"""Exercise upload boundaries against real temporary trees and ZIP files."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools.prepare_repository import (
    Finding, PreparationError, Snapshot, export_snapshot, inspect_repository,
    output_path, render_gitignore, safe_relative, scan_text, validate_policy,
)


def policy():
    return {
        "version": 1,
        "root_files": [".gitignore", ".env.example", "README.md", "app.py"],
        "directory_rules": [
            {"path": "tests", "patterns": ["*.py", "*.md"], "recursive": False},
            {"path": "course_analysis", "patterns": ["*.md"], "recursive": True},
        ],
        "baseline_files": ["baselines/required.txt", "baselines/boundary/templates.json"],
        "max_file_bytes": 1024 * 1024,
        "max_total_bytes": 10 * 1024 * 1024,
    }


class RepositoryPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.policy = policy()
        for name, content in {
            ".gitignore": render_gitignore(self.policy),
            ".env.example": "GENERATOR_A_API_KEY=\n",
            "README.md": "[Baseline](baselines/required.txt)\n",
            "app.py": "print('example')\n",
            "tests/test_app.py": "# offline test\n",
            "tests/LEGACY_FIXTURES.md": "# fixture description\n",
            "course_analysis/focus/NOTES.md": "# course conclusions\n",
            "baselines/required.txt": "comparison template\n",
            "baselines/boundary/templates.json": '{"template": "reference"}\n',
        }.items():
            self.write(name, content)

    def write(self, name, contents):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")

    def inspect(self):
        return inspect_repository(self.root, self.policy)

    def directory_link(self, linked, target):
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(target), str(linked))
        else:
            linked.symlink_to(target, target_is_directory=True)
        self.addCleanup(lambda: linked.rmdir() if os.name == "nt" else linked.unlink())

    def test_private_files_are_not_read_hashed_or_exported(self):
        for name in (".env", ".env.backup", "tests/.env", "reports/private.md",
                     "maintenance/log.json", "baselines/history/private.txt",
                     "evals/results/private.json", "evals/artifacts/private.txt",
                     "last_optimized_prompt.md", ".venv/private.py",
                     "optimizer_settings.json", "tests/optimizer_settings.json",
                     "context_windows.json", "tests/context_windows.json"):
            self.write(name, "private sentinel")
        seen = []
        original = Path.read_bytes

        def reading(path):
            relative = path.relative_to(self.root).as_posix()
            seen.append(relative)
            self.assertFalse(relative.startswith(".env") and relative != ".env.example")
            self.assertNotEqual(path.name, ".env")
            self.assertNotEqual(path.name, "optimizer_settings.json")
            self.assertNotEqual(path.name, "context_windows.json")
            return original(path)

        with patch.object(Path, "read_bytes", reading):
            snapshot = self.inspect()
        self.assertFalse(snapshot.findings)
        self.assertTrue(snapshot.env_exists)
        self.assertEqual(set(seen), set(snapshot.files))
        archive = output_path(self.root, "maintenance/source.zip", ".zip")
        export_snapshot(snapshot, archive)
        with zipfile.ZipFile(archive) as package:
            self.assertEqual(set(package.namelist()), set(snapshot.files))
            self.assertNotIn(b"private sentinel", b"".join(package.read(name) for name in package.namelist()))
            self.assertIn("baselines/boundary/templates.json", package.namelist())

    def test_window_template_is_exported_without_reading_local_bindings(self):
        self.policy["root_files"].append("context_windows.example.json")
        self.write(".gitignore", render_gitignore(self.policy))
        template = '{"version":1,"roles":{}}\n'
        (self.root / "context_windows.example.json").write_bytes(template.encode())
        self.write("context_windows.json", "private window sentinel")
        original = Path.read_bytes

        def reading(path):
            self.assertNotEqual(path.name.casefold(), "context_windows.json")
            return original(path)

        with patch.object(Path, "read_bytes", reading):
            snapshot = self.inspect()
        self.assertFalse(snapshot.findings)
        archive = output_path(self.root, "maintenance/source.zip", ".zip")
        export_snapshot(snapshot, archive)
        with zipfile.ZipFile(archive) as package:
            self.assertNotIn("context_windows.json", package.namelist())
            self.assertEqual(package.read("context_windows.example.json"), template.encode())

    def test_window_bindings_cannot_be_added_to_the_public_policy(self):
        self.policy["root_files"].append("context_windows.json")
        with self.assertRaisesRegex(PreparationError, "protected-policy-path"):
            validate_policy(self.policy)

    def test_required_baseline_missing_blocks_export(self):
        (self.root / "baselines/required.txt").unlink()
        snapshot = self.inspect()
        self.assertIn(Finding("baselines/required.txt", 0, "missing-source-file"), snapshot.findings)
        with self.assertRaisesRegex(PreparationError, "export-blocked"):
            export_snapshot(snapshot, self.root / "maintenance/source.zip")
        self.assertFalse((self.root / "maintenance/source.zip").exists())

    def test_finding_output_never_contains_credential_or_personal_path(self):
        credential = "sk-" + "A" * 32
        personal = "/".join(("C:", "Users", "Someone", "private"))
        text = credential + "\n" + personal + "\n[Private report](reports/private.md)\n"
        findings = scan_text("README.md", text, {"README.md"})
        self.assertEqual({item.risk for item in findings}, {
            "credential-token", "absolute-personal-path", "unpublished-local-link"})
        rendered = json.dumps([item.to_dict() for item in findings])
        self.assertNotIn(credential, rendered)
        self.assertNotIn(personal, rendered)

    def test_env_example_must_not_contain_real_credential(self):
        value = "GENERATOR_A_API_KEY=" + "example-nonempty"
        self.write(".env.example", value)
        self.assertIn(Finding(".env.example", 1, "nonempty-example-credential"), self.inspect().findings)

    def test_only_exact_known_test_dummy_literal_is_exempt(self):
        source = (Path(__file__).parent / "test_review.py").read_text(encoding="utf-8-sig")
        self.assertNotIn("literal-credential", {item.risk for item in scan_text("tests/test_review.py", source, set())})
        self.assertIn("literal-credential", {item.risk for item in scan_text("tests/another_test.py", source, set())})
        replaced = "api_key='" + "sk-" + "B" * 32 + "'"
        self.assertEqual({item.risk for item in scan_text("tests/test_review.py", replaced, set())},
                         {"credential-token", "literal-credential"})
        header = "-----BEGIN " + "PRIVATE KEY-----"
        self.assertEqual({item.risk for item in scan_text("tests/test_review.py", header, set())}, {"private-key"})

    def test_parent_paths_and_protected_policy_paths_are_rejected(self):
        for name in ("../.env", "C:/secret.txt", "tests/../app.py", "/app.py"):
            with self.subTest(name=name), self.assertRaises(PreparationError):
                safe_relative(name)
        for name in (".env", ".env.backup", "tests/.env", "reports/private.md", "optimizer_settings.json"):
            changed = policy()
            changed["root_files"] = [name]
            with self.subTest(name=name), self.assertRaises(PreparationError):
                validate_policy(changed)

    def test_outputs_cannot_overwrite_sources_or_escape_maintenance(self):
        for name in (".env", "app.py", "../source.zip", "maintenance/../.env", "maintenance/../../source.zip"):
            with self.subTest(name=name), self.assertRaises(PreparationError):
                output_path(self.root, name, ".zip")
        allowed = output_path(self.root, "maintenance/20261006/source.zip", ".zip")
        self.assertEqual(allowed, self.root / "maintenance/20261006/source.zip")

    def test_relative_output_under_an_aliased_root_uses_the_canonical_root(self):
        alias = self.root / "root-alias"
        self.directory_link(alias, self.root)
        allowed = output_path(alias, "maintenance/source.zip", ".zip")
        self.assertEqual(allowed, self.root / "maintenance/source.zip")
        for name in ("maintenance/../../source.zip", "app.py", ".env"):
            with self.subTest(name=name), self.assertRaises(PreparationError):
                output_path(alias, name, ".zip")

    def test_source_hardlink_to_private_config_is_never_read(self):
        self.write(".env", "private hardlink sentinel")
        alias = self.root / "tests/alias.py"
        os.link(self.root / ".env", alias)
        snapshot = self.inspect()
        self.assertIn(Finding("tests/alias.py", 0, "hardlinked-source-file"), snapshot.findings)
        self.assertNotIn("tests/alias.py", snapshot.files)

    def test_directory_link_is_not_followed(self):
        target = self.root / "private"
        self.write("private/private.md", "private linked sentinel")
        linked = self.root / "course_analysis/linked"
        self.directory_link(linked, target)
        snapshot = self.inspect()
        self.assertIn(Finding("course_analysis/linked", 0, "linked-path"), snapshot.findings)
        self.assertNotIn("course_analysis/linked/private.md", snapshot.files)

    def test_linked_output_directory_is_rejected(self):
        (self.root / "outside").mkdir()
        linked = self.root / "maintenance"
        self.directory_link(linked, self.root / "outside")
        with self.assertRaisesRegex(PreparationError, "linked-output"):
            output_path(self.root, "maintenance/source.zip", ".zip")

    def test_zip_uses_the_scanned_snapshot_even_if_source_changes(self):
        snapshot = self.inspect()
        previous = snapshot.files["app.py"]
        self.write("app.py", "private replacement after scan")
        first = output_path(self.root, "maintenance/source.zip", ".zip")
        second = output_path(self.root, "maintenance/second.zip", ".zip")
        export_snapshot(snapshot, first)
        export_snapshot(snapshot, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as package:
            self.assertEqual(package.read("app.py"), previous)

    def test_changed_ignore_and_unlisted_root_source_require_review(self):
        self.write(".gitignore", "*.py\n")
        self.write("new.py", "# omitted module\n")
        findings = self.inspect().findings
        self.assertIn(Finding(".gitignore", 0, "gitignore-policy-mismatch"), findings)
        self.assertIn(Finding("new.py", 0, "unlisted-source-file"), findings)

    def test_genuine_git_ignore_and_index_match_upload_policy(self):
        self.write(".env", "private sentinel")
        self.write("tests/__pycache__/private.py", "private sentinel")
        self.write("context_windows.json", "private window sentinel")
        self.write("baselines/history/private.txt", "private sentinel")
        subprocess.run(["git", "init", "-q"], cwd=self.root, capture_output=True, check=True)
        subprocess.run(["git", "add", "."], cwd=self.root, capture_output=True, check=True)
        tracked = subprocess.run(["git", "ls-files", "-z"], cwd=self.root, capture_output=True, check=True)
        snapshot = self.inspect()
        self.assertFalse(snapshot.findings)
        self.assertEqual(set(filter(None, tracked.stdout.decode().split("\0"))), set(snapshot.files))
        subprocess.run(["git", "add", "-f", ".env"], cwd=self.root, capture_output=True, check=True)
        self.assertIn(Finding(".env", 0, "tracked-outside-upload-policy"), self.inspect().findings)
        subprocess.run(["git", "add", "-f", "context_windows.json"], cwd=self.root, capture_output=True, check=True)
        self.assertIn(Finding("context_windows.json", 0, "tracked-outside-upload-policy"), self.inspect().findings)

    def test_manually_mutated_export_cannot_include_private_or_parent_paths(self):
        for name in (".env", "../outside.txt"):
            with self.subTest(name=name), self.assertRaises(PreparationError):
                export_snapshot(Snapshot(files={name: b"private"}), self.root / "maintenance/source.zip")


if __name__ == "__main__":
    unittest.main()
