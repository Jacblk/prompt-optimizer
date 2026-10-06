"""Local dialogue reports and guarded saves; importing this never loads .env."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re

from optimizer_config import ConfigurationError, ROOT
from optimizer_handoff import WindowLimits, model_identity, read_window_config, required_roles
from optimizer_io import atomic_write


@dataclass
class SaveOutcome:
    report_path: Path | None = None
    prompt_path: Path | None = None
    report_saved: bool = False
    prompt_saved: bool = False
    errors: list[str] = field(default_factory=list)


def _same_path(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        raise ConfigurationError("无法核对保存路径，未写入文件。") from None


def _project_protected(root: Path):
    names = (".env", "context_windows.json", "requirements.txt", "requirements-tui.txt",
             "README.md", "TUI_WORKFLOW.md", "TUI_ACCEPTANCE.md", "tui.tcss", "HANDOFF_WORKFLOW.md", "HANDOFF_ACCEPTANCE.md",
             "COMPATIBILITY.md", "CLI_WORKFLOW.md", "REFERENCE_FILES.md", "IMPROVEMENT_PLAN.md",
             "LANGCHAIN_REUSE.md", "NATURAL_MATERIAL_REFERENCES.md", "启动提示词优化器.cmd", "prompt-optimizer.cmd")
    return ([root / name for name in names] + list(root.glob("*.py"))
            + list((root / "tests").glob("*.py")))


def _validate_destinations(destinations, protected):
    targets = [Path(p).expanduser().resolve() for p in destinations]
    sources = [Path(p).expanduser().resolve() for p in protected if p is not None]
    if any(p.name.casefold().startswith(".env") for p in targets):
        raise ConfigurationError("保存路径不能覆盖配置文件。")
    if any(_same_path(a, b) for index, a in enumerate(targets) for b in targets[index + 1:]):
        raise ConfigurationError("报告和提示词保存路径不能重合。")
    if any(_same_path(a, b) for a in targets for b in sources):
        raise ConfigurationError("保存路径不能覆盖输入、参考、配置或项目源文件。")
    return targets


def _result_sources(data):
    metadata = data.get("metadata") or {}
    paths = []
    for key, item_key in (("reference_files", "files"), ("handoff", "sources")):
        section = metadata.get(key) or {}
        for source in section.get(item_key, []):
            value = source.get("source")
            if isinstance(value, str) and value:
                paths.append(Path(value))
    return paths


def validate_dialogue_paths(*, root=ROOT, report_path=None, output_path=None, protected_paths=()):
    """Check requested destinations before starting paid work; saves recheck them."""
    root = Path(root).expanduser().resolve()
    output = output_path if output_path is not None else root / "last_optimized_prompt.md"
    destinations = [output] + ([report_path] if report_path is not None else [])
    targets = _validate_destinations(destinations, _project_protected(root) + list(protected_paths))
    for target in targets:
        if target.is_dir() or any(parent.exists() and not parent.is_dir() for parent in target.parents):
            raise ConfigurationError("保存路径须为文件，且父路径不能是已有文件。")
    if report_path is not None:
        _validate_destinations([report_path], [root / "last_optimized_prompt.md"])


def save_dialogue_result(result, *, root=ROOT, current=None, report_path=None,
                         output_path=None, protected_paths=()) -> SaveOutcome:
    """Always save a checkpoint; publish only when the current-version gate passes.

    ``current`` is a synchronous callable accepting the result. Without it this is
    a report-only save, including when an old successful report is being viewed.
    Errors are safe display strings; saving never changes the model result.
    """
    root = Path(root).expanduser().resolve()
    data = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    dialogue = (data.get("metadata") or {}).get("dialogue") or {}
    label = re.sub(r"[^a-zA-Z0-9_-]", "", str(dialogue.get("session_id", "preparation")))[:48] or "dialogue"
    revision = dialogue.get("revision", 0)
    if type(revision) is not int or revision < 0:
        revision = 0
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    report_path = Path(report_path) if report_path is not None else root / "reports" / f"{stamp}-{label}-r{revision}.json"
    output_path = Path(output_path) if output_path is not None else root / "last_optimized_prompt.md"
    outcome = SaveOutcome(report_path=report_path.expanduser().resolve(),
                          prompt_path=output_path.expanduser().resolve())
    protected = _project_protected(root) + list(protected_paths) + _result_sources(data)
    try:
        _validate_destinations([outcome.report_path, outcome.prompt_path], protected)
        _validate_destinations([outcome.report_path], [root / "last_optimized_prompt.md"])
    except (ConfigurationError, OSError, ValueError):
        outcome.errors.append("保存路径冲突或不可用，未写入文件。")
        return outcome
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        atomic_write(outcome.report_path, text)
        outcome.report_saved = True
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        outcome.errors.append("报告保存失败；界面中的需求与结果仍可保存或重试。")
    published = publish_dialogue_prompt(result, root=root, current=current,
                                        output_path=outcome.prompt_path, protected_paths=protected_paths)
    outcome.prompt_saved = published.prompt_saved
    outcome.errors.extend(published.errors)
    return outcome


def publish_dialogue_prompt(result, *, root=ROOT, current=None, output_path=None,
                            protected_paths=()) -> SaveOutcome:
    """Publish on the session-owning event loop, serialized with input changes.

    Report serialization may run in a worker thread with ``current=None``. The
    final gate and short atomic prompt write must run synchronously on the loop
    which changes the session/input; a gate inside a worker cannot guard edits.
    """
    root = Path(root).expanduser().resolve()
    data = ({"status": result.status, "optimized_prompt": result.optimized_prompt,
             "metadata": result.metadata} if hasattr(result, "status") else dict(result))
    output_path = Path(output_path) if output_path is not None else root / "last_optimized_prompt.md"
    outcome = SaveOutcome(prompt_path=output_path.expanduser().resolve())
    if data.get("status") not in {"ready", "unreviewed"}:
        return outcome
    try:
        _validate_destinations([outcome.prompt_path],
                               _project_protected(root) + list(protected_paths) + _result_sources(data))
    except (ConfigurationError, OSError, ValueError):
        outcome.errors.append("保存路径冲突或不可用，未写入提示词。")
        return outcome
    prompt = data.get("optimized_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        outcome.errors.append("成功状态没有有效提示词，未更新成功文件。")
        return outcome
    try:
        eligible = current is not None and bool(current(result))
    except Exception:
        eligible = False
        outcome.errors.append("无法核对当前会话版本，未更新成功提示词。")
    if not eligible:
        return outcome
    try:
        # The caller serializes this synchronous block with all version changes.
        if not current(result):
            return outcome
        atomic_write(outcome.prompt_path, prompt)
        outcome.prompt_saved = True
    except (OSError, UnicodeError):
        outcome.errors.append("提示词保存失败；界面中的结果仍可复制或重试。")
    except Exception:
        outcome.errors.append("无法核对当前会话版本，未更新成功提示词。")
    return outcome


def save_draft(text: str, path: Path, *, protected_paths=()) -> Path:
    """Explicit draft export never replaces the automatic successful prompt."""
    if not isinstance(text, str) or not text.strip():
        raise ConfigurationError("没有可保存的草稿。")
    path = Path(path).expanduser().resolve()
    protected = (_project_protected(ROOT) + _project_protected(path.parent)
                 + [ROOT / "last_optimized_prompt.md", path.parent / "last_optimized_prompt.md"]
                 + list(protected_paths))
    _validate_destinations([path], protected)
    atomic_write(path, "# 待确认草稿\n\n" + text)
    return path


def load_report(path: Path) -> dict:
    """Read old/new reports offline; a report is display data, never a replay."""
    path = Path(path).expanduser().resolve()
    if path.name.casefold().startswith(".env"):
        raise ConfigurationError("配置文件不能作为报告读取。")
    try:
        with path.open("rb") as stream:
            raw = stream.read(32 * 1024 * 1024 + 1)
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError
        def unique(pairs):
            data = {}
            for key, value in pairs:
                if key in data:
                    raise ValueError
                data[key] = value
            return data
        def reject_constant(value):
            raise ValueError
        data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=reject_constant)
        if not isinstance(data, dict) or not isinstance(data.get("status"), str):
            raise ValueError
        if (data.get("metadata") is not None and not isinstance(data["metadata"], dict)
                or data.get("optimized_prompt") is not None and not isinstance(data["optimized_prompt"], str)):
            raise ValueError
        for key in ("candidates", "reviews"):
            if key in data and (not isinstance(data[key], list) or any(not isinstance(item, dict) for item in data[key])):
                raise ValueError
        metadata = data.get("metadata") or {}
        if metadata.get("dialogue") is not None and not isinstance(metadata["dialogue"], dict):
            raise ValueError
        return data
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise ConfigurationError("报告无法读取或格式不完整。") from None


def configure_dialogue_windows(configs, values, *, path=ROOT / "context_windows.json") -> dict:
    """Validate a complete modal response, then atomically save public bindings."""
    path = Path(path).expanduser().resolve()
    if not isinstance(values, dict):
        raise ConfigurationError("窗口填写结构不正确，未修改配置。")
    if set(values) != set(required_roles()):
        raise ConfigurationError("请填写生成器 A、B 和评审 C 的全部窗口，未修改配置。")
    _validate_destinations([path], [p for p in _project_protected(path.parent)
                                  if p.name != "context_windows.json"]
                           + [path.parent / "last_optimized_prompt.md"])
    old = read_window_config(path) if path.exists() else {"version": 1, "roles": {}}
    fields = {"model", "service_sha256", "identity_sha256", "context_window"}
    roles = {role: {k: v for k, v in entry.items() if k in fields}
             for role, entry in old["roles"].items()
             if role in set(required_roles()) and isinstance(entry, dict)}
    for role in required_roles():
        if role not in configs:
            raise ConfigurationError("模型角色配置不完整，未修改窗口配置。")
        raw = values[role]
        try:
            if type(raw) is not int and not isinstance(raw, str):
                raise ValueError
            size = int(raw.strip()) if isinstance(raw, str) else raw
            if size < 1:
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise ConfigurationError(f"{role} 窗口须为正整数，未修改配置。") from None
        roles[role] = {**model_identity(configs[role]), "context_window": size}
    result = {"version": 1, "roles": roles}
    WindowLimits.from_config(result, configs)
    atomic_write(path, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result
