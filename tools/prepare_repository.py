"""Read-only upload preflight; export only the reviewed source allowlist.

This tool never reads, scans, hashes, or copies private environment files. Its
only observation about the project's .env is whether that directory entry exists.
Finding output contains file names, line numbers, and risk types, never matches.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import subprocess
import sys
import tempfile
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = "tools/repository_policy.json"
PRIVATE_DIRECTORIES = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "reports", "maintenance", "node_modules",
})
SOURCE_SUFFIXES = frozenset({".py", ".md", ".json", ".jsonl", ".txt", ".cmd", ".tcss", ".yml", ".yaml"})
EXCLUDED_DESCRIPTION = [
    ".env and private .env variants (existence only; contents never accessed)",
    "virtual environments and caches", "reports/", "last_optimized_prompt.md",
    "optimizer_settings.json", "context_windows.json",
    "evals/results/", "evals/artifacts/", "maintenance/", "unlisted baseline archives",
    "all other paths absent from the upload policy",
]
SECRET_PATTERNS = (
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("credential-token", re.compile(r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16})\b")),
    ("literal-credential", re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*['\"](?P<value>[A-Za-z0-9_+/=-]{24,})['\"]")),
    ("absolute-personal-path", re.compile(r"(?i)(?:[a-z]:[\\/]+Users[\\/]+[^\s<>\"')]+|/(?:home|Users)/[^\s<>\"')]+)")),
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]\n]*\]\((<[^>\n]+>|[^\s)]+)(?:\s+['\"][^\n]*?['\"])?\)")
# Explicit dummy credential in the strict-parser/redaction test, identified by
# its exact value hash. No other literals or credential-token rules are exempt.
TEST_DUMMY_FINGERPRINTS = {
    "tests/test_review.py": frozenset({"718186fef2a10c2ee570214de2caf2758b93d7aff2a0837db000c0532f2c7a8c"}),
}


class PreparationError(ValueError):
    """A safe error whose message never includes source contents."""


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    risk: str

    def to_dict(self):
        return {"path": self.path, "line": self.line, "risk": self.risk}


@dataclass
class Snapshot:
    files: dict[str, bytes] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    env_exists: bool = False

    def manifest(self):
        return {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "policy": POLICY_PATH,
            "env_exists": self.env_exists,
            "env_content_accessed": False,
            "file_count": len(self.files),
            "total_bytes": sum(map(len, self.files.values())),
            "files": [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                      for name, data in sorted(self.files.items())],
            "findings": [finding.to_dict() for finding in sorted(set(self.findings))],
            "excluded": EXCLUDED_DESCRIPTION,
            "license_selected": False,
            "review_note": "Pattern scanning does not replace review of the manifest and source contents.",
        }


def safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PreparationError("invalid-policy-path")
    path = PurePosixPath(value)
    if path.is_absolute() or ":" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise PreparationError("invalid-policy-path")
    return path.as_posix()


def forbidden(name: str) -> bool:
    parts = PurePosixPath(name).parts
    lower = tuple(part.casefold() for part in parts)
    if name == ".env.example":
        return False
    return (any(part.startswith(".env") for part in lower)
            or any(part in PRIVATE_DIRECTORIES for part in lower)
            or lower[-1] in {"last_optimized_prompt.md", "optimizer_settings.json", "context_windows.json"}
            or lower[-1].startswith(".optimizer-")
            or lower[-1].endswith((".pyc", ".pyo"))
            or lower[:2] in {("evals", "results"), ("evals", "artifacts")})


def checked_path(root: Path, name: str) -> Path:
    name = safe_relative(name)
    if forbidden(name):
        raise PreparationError("protected-path")
    current = root
    for part in PurePosixPath(name).parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(metadata.st_mode)
                or getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise PreparationError("linked-path")
        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink > 1:
            raise PreparationError("hardlinked-source-file")
    resolved_root = root.resolve()
    if not current.resolve().is_relative_to(resolved_root):
        raise PreparationError("outside-repository")
    return current


def validate_policy(policy: dict):
    if policy.get("version") != 1:
        raise PreparationError("unsupported-policy")
    names = policy.get("root_files", []) + policy.get("baseline_files", [])
    if not names or len(names) != len(set(name.casefold() for name in names)):
        raise PreparationError("duplicate-or-empty-policy")
    for name in names:
        safe_relative(name)
        if forbidden(name):
            raise PreparationError("protected-policy-path")
    if any("/" in name for name in policy["root_files"]):
        raise PreparationError("invalid-root-policy-path")
    for name in policy["baseline_files"]:
        if not name.startswith("baselines/"):
            raise PreparationError("invalid-baseline-policy-path")
    for rule in policy.get("directory_rules", []):
        safe_relative(rule["path"])
        if forbidden(rule["path"]):
            raise PreparationError("protected-policy-directory")
        for pattern in rule["patterns"]:
            if not pattern or "/" in pattern or "\\" in pattern or ".." in pattern or ":" in pattern:
                raise PreparationError("invalid-policy-pattern")
    for key in ("max_file_bytes", "max_total_bytes"):
        if type(policy.get(key)) is not int or policy[key] < 1:
            raise PreparationError("invalid-policy-size-limit")


def load_policy(root: Path) -> dict:
    path = checked_path(root, POLICY_PATH)
    try:
        policy = json.loads(path.read_text(encoding="utf-8-sig"))
        validate_policy(policy)
        return policy
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
        raise PreparationError("invalid-or-unreadable-policy") from error


def render_gitignore(policy: dict) -> str:
    validate_policy(policy)
    lines = ["# Generated from tools/repository_policy.json. Review the policy before adding files.", "/*"]
    lines.extend("!/" + name for name in policy["root_files"])
    for rule in policy["directory_rules"]:
        parts = PurePosixPath(rule["path"]).parts
        for index in range(1, len(parts) + 1):
            parent = "/" + "/".join(parts[:index]) + "/"
            if "!" + parent not in lines:
                lines.extend(["!" + parent, parent + "*"])
        base = "/" + rule["path"] + "/"
        if rule["recursive"]:
            lines.extend([base + "**", "!" + base + "**/"])
            lines.extend("!" + base + "**/" + pattern for pattern in rule["patterns"])
        else:
            lines.extend("!" + base + pattern for pattern in rule["patterns"])
    lines.extend(["!/baselines/", "/baselines/*"])
    for name in policy["baseline_files"]:
        parts = PurePosixPath(name).parts
        for index in range(2, len(parts)):
            parent = "/" + "/".join(parts[:index]) + "/"
            if "!" + parent not in lines:
                lines.extend(["!" + parent, parent + "*"])
        lines.append("!/" + name)
    lines.extend([
        "", "# Private data and generated artifacts remain excluded at every depth.",
        ".env*", "!/.env.example", ".venv/", "venv/", "__pycache__/", "*.py[cod]",
        ".pytest_cache/", ".mypy_cache/", ".ruff_cache/", ".coverage", "htmlcov/",
        "node_modules/", "reports/", "/evals/results/", "/evals/artifacts/", "/maintenance/",
        "last_optimized_prompt.md", "optimizer_settings.json", "context_windows.json",
        ".optimizer-*", ".DS_Store", "Thumbs.db", "",
    ])
    return "\n".join(lines)


def collect_candidates(root: Path, policy: dict, snapshot: Snapshot) -> list[str]:
    names = set(policy["root_files"] + policy["baseline_files"])
    for rule in policy["directory_rules"]:
        try:
            directory = checked_path(root, rule["path"])
        except PreparationError as error:
            snapshot.findings.append(Finding(rule["path"], 0, str(error)))
            continue
        if not directory.is_dir():
            snapshot.findings.append(Finding(rule["path"], 0, "missing-source-directory"))
            continue
        for current, directories, files in os.walk(directory, followlinks=False):
            safe_directories = []
            for child in directories:
                relative = (Path(current) / child).relative_to(root).as_posix()
                if forbidden(relative):
                    continue
                try:
                    checked_path(root, relative)
                except PreparationError as error:
                    snapshot.findings.append(Finding(relative, 0, str(error)))
                else:
                    safe_directories.append(child)
            directories[:] = safe_directories if rule["recursive"] else []
            for file in files:
                relative = (Path(current) / file).relative_to(root).as_posix()
                if not forbidden(relative) and any(fnmatch.fnmatchcase(file, pattern) for pattern in rule["patterns"]):
                    names.add(relative)
    for entry in root.iterdir():
        if (not forbidden(entry.name) and entry.suffix.lower() in SOURCE_SUFFIXES
                and entry.name not in names and not entry.is_dir()):
            snapshot.findings.append(Finding(entry.name, 0, "unlisted-source-file"))
    return sorted(names)


def scan_text(name: str, text: str, candidate_names: set[str]) -> list[Finding]:
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        for risk, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(line):
                if (risk == "literal-credential" and hashlib.sha256(match.group("value").encode()).hexdigest()
                        in TEST_DUMMY_FINGERPRINTS.get(name, ())):
                    continue
                findings.append(Finding(name, number, risk))
        if name == ".env.example" and re.match(r"\s*[A-Z0-9_]*(?:API_KEY|SECRET|PASSWORD|TOKEN)\s*=\s*[^\s#]", line):
            # MAX_TOKENS and TOKEN_LIMIT_FIELD are not credential variables.
            key = line.split("=", 1)[0].strip()
            if not key.endswith(("MAX_TOKENS", "TOKEN_LIMIT_FIELD")):
                findings.append(Finding(name, number, "nonempty-example-credential"))
        if name.endswith(".md"):
            for match in MARKDOWN_LINK.finditer(line):
                target = unquote(match.group(1).strip("<>"))
                if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith(("#", "//")):
                    continue
                target = target.split("#", 1)[0].split("?", 1)[0]
                if target:
                    relative = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
                    if relative not in candidate_names:
                        findings.append(Finding(name, number, "unpublished-local-link"))
    return findings


def inspect_repository(root: Path, policy: dict | None = None) -> Snapshot:
    root = root.resolve()
    policy = policy if policy is not None else load_policy(root)
    validate_policy(policy)
    snapshot = Snapshot(env_exists=os.path.lexists(root / ".env"))
    candidates = collect_candidates(root, policy, snapshot)
    for name in candidates:
        try:
            path = checked_path(root, name)
            if not path.is_file():
                snapshot.findings.append(Finding(name, 0, "missing-source-file"))
                continue
            if path.stat().st_size > policy["max_file_bytes"]:
                snapshot.findings.append(Finding(name, 0, "oversized-source-file"))
                continue
            data = path.read_bytes()
            if len(data) > policy["max_file_bytes"]:
                snapshot.findings.append(Finding(name, 0, "oversized-source-file"))
                continue
            snapshot.files[name] = data
            text = data.decode("utf-8-sig")
        except PreparationError as error:
            snapshot.findings.append(Finding(name, 0, str(error)))
            continue
        except (OSError, UnicodeError):
            snapshot.findings.append(Finding(name, 0, "unreadable-or-nontext-source"))
            continue
        snapshot.findings.extend(scan_text(name, text, set(candidates)))
    if sum(map(len, snapshot.files.values())) > policy["max_total_bytes"]:
        snapshot.findings.append(Finding(".", 0, "oversized-source-package"))
    expected = render_gitignore(policy).splitlines()
    actual = snapshot.files.get(".gitignore", b"").decode("utf-8-sig").splitlines()
    if actual != expected:
        snapshot.findings.append(Finding(".gitignore", 0, "gitignore-policy-mismatch"))
    if (root / ".git").exists():
        try:
            result = subprocess.run(["git", "ls-files", "--cached", "-z"], cwd=root,
                                    capture_output=True, check=True)
            tracked = result.stdout.decode("utf-8").split("\0")
            for name in filter(None, tracked):
                if name not in snapshot.files or forbidden(name):
                    snapshot.findings.append(Finding(name, 0, "tracked-outside-upload-policy"))
        except (OSError, UnicodeError, subprocess.CalledProcessError):
            snapshot.findings.append(Finding(".git", 0, "git-index-check-failed"))
    snapshot.findings = sorted(set(snapshot.findings))
    return snapshot


def output_path(root: Path, value: str | Path, suffix: str) -> Path:
    # Windows temporary roots may use an 8.3 alias; expand relative output
    # against the same canonical root used by containment checks.
    root = root.resolve()
    requested = Path(value)
    if not requested.is_absolute():
        requested = root / requested
    # Output is confined to local maintenance, never a source or config file.
    try:
        relative = requested.absolute().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise PreparationError("output-outside-repository") from error
    parts = PurePosixPath(relative).parts
    if len(parts) < 2 or parts[0] != "maintenance" or ".." in parts or requested.suffix != suffix:
        raise PreparationError("output-must-be-maintenance-file")
    current = root.resolve()
    for part in parts:
        current /= part
        if current.is_symlink() or (current.exists() and getattr(current.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise PreparationError("linked-output-path")
        if current.exists() and current.is_file() and current.stat().st_nlink > 1:
            raise PreparationError("hardlinked-output-path")
    if not current.resolve().is_relative_to((root / "maintenance").resolve()):
        raise PreparationError("output-outside-maintenance")
    return current


def export_snapshot(snapshot: Snapshot, path: Path):
    if snapshot.findings:
        raise PreparationError("export-blocked-by-findings")
    for name in snapshot.files:
        safe_relative(name)
        if forbidden(name):
            raise PreparationError("protected-export-path")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".repository-", suffix=".zip", dir=path.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, contents in sorted(snapshot.files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, contents)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_manifest(snapshot: Snapshot, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot.manifest(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only upload preflight; private .env files are never read.")
    parser.add_argument("--manifest", help="Save a relative-path JSON inventory under maintenance/.")
    parser.add_argument("--archive", help="Export reviewed source files to a ZIP under maintenance/; blocks on findings.")
    args = parser.parse_args(argv)
    try:
        manifest_path = output_path(ROOT, args.manifest, ".json") if args.manifest else None
        archive_path = output_path(ROOT, args.archive, ".zip") if args.archive else None
        snapshot = inspect_repository(ROOT)
        if manifest_path:
            save_manifest(snapshot, manifest_path)
        if archive_path:
            export_snapshot(snapshot, archive_path)
        result = {"files": len(snapshot.files), "bytes": sum(map(len, snapshot.files.values())),
                  "env_exists": snapshot.env_exists, "env_content_accessed": False,
                  "findings": [finding.to_dict() for finding in snapshot.findings]}
        if archive_path:
            result["archive"] = archive_path.relative_to(ROOT).as_posix()
        if manifest_path:
            result["manifest"] = manifest_path.relative_to(ROOT).as_posix()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if snapshot.findings else 0
    except (PreparationError, OSError) as error:
        # Never serialize an OS error, which can contain unreviewed contents/paths.
        risk = str(error) if isinstance(error, PreparationError) else "preparation-io-error"
        print(json.dumps({"findings": [{"path": ".", "line": 0, "risk": risk}]}))
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
