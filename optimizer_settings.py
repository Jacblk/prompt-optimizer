"""Shared offline configuration schema and transactions for CLI and Textual.

Only explicit configuration actions instantiate/load ConfigurationStore. Import,
runtime defaults, material preview and help never read model credentials.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import stat
from urllib.parse import urlsplit

from optimizer_config import (
    ROOT, ConfigurationError, REASONING_EFFORTS, handoff_model_config,
    model_config,
)
from optimizer_io import atomic_write_bytes

SETTINGS_NAME = "optimizer_settings.json"
ROLE_PREFIXES = {"a": "GENERATOR_A", "b": "GENERATOR_B", "c": "JUDGE"}
INTERNAL_ROLES = {"a": "a", "b": "b", "c": "judge"}
UNSET = object()


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    default: object = None
    kind: str = "int"
    minimum: float = 1
    maximum: float | None = None
    choices: tuple = ()
    optional: bool = False
    advanced: bool = False
    secret: bool = False
    help: str = ""
    suffix: str | None = None

    def parse(self, raw):
        if raw is None or isinstance(raw, str) and (not raw.strip() or self.optional and raw.strip() == "null"):
            if self.optional:
                return None
            if self.kind == "str":
                return ""
            raise ConfigurationError(f"{self.key} 不能为空。")
        try:
            if self.kind == "int":
                if isinstance(raw, bool) or not isinstance(raw, (int, str)):
                    raise ValueError
                value = int(raw)
            elif self.kind == "float":
                if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
                    raise ValueError
                value = float(raw)
            elif self.kind == "bool":
                if type(raw) is bool:
                    value = raw
                elif isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
                    value = raw.strip().lower() == "true"
                else:
                    raise ValueError
            elif self.kind == "json":
                value = strict_json(raw) if isinstance(raw, str) else deepcopy(raw)
                if not isinstance(value, dict):
                    raise ValueError
                json.dumps(value, allow_nan=False)
            else:
                if not isinstance(raw, str) or any(c in raw for c in "\r\n\x00"):
                    raise ValueError
                value = raw.strip()
            if self.kind in {"int", "float"}:
                if not math.isfinite(value) or value < self.minimum:
                    raise ValueError
                if self.maximum is not None and value > self.maximum:
                    raise ValueError
            if self.choices and value not in self.choices:
                raise ValueError
            return value
        except (ValueError, TypeError, OverflowError, RecursionError):
            raise ConfigurationError(f"{self.key} 的类型、范围或选项无效。") from None


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    def constant(value):
        raise ValueError

    return json.loads(text, object_pairs_hook=unique, parse_constant=constant)


_runtime = [
    Setting("run.network_max_attempts", "网络失败最大尝试数", 2,
            help="含首次请求；流式兼容回退另允许一次。"),
    Setting("run.handoff_max_attempts", "历史整理最大尝试数", 2,
            help="每个阶段分别计算，包含整理与独立核验。"),
    Setting("run.schema_max_attempts", "澄清／分层结构最大尝试数", 2),
    Setting("run.prompt_max_repairs", "每会话提示词最大修复次数", 1, minimum=0,
            help="0 关闭；修复消耗在同一会话持续累计。"),
    Setting("run.token_budget", "默认会话 token 预算", optional=True,
            help="留空不设预算；TUI 或 CLI 可为本次会话覆盖。"),
    Setting("run.max_input_chars", "需求字符上限", optional=True, advanced=True),
    Setting("run.max_builtin_examples", "内置示例数量上限", 3, minimum=0, advanced=True),
    Setting("run.max_example_chars", "内置示例字符预算", 4000, minimum=0, advanced=True),
    Setting("history.chunk_bytes", "历史分块字节数", 24000, advanced=True),
    Setting("history.max_files", "历史文件数量上限", 10, maximum=10, advanced=True),
    Setting("history.max_file_bytes", "历史单文件字节上限", 5242880, maximum=5242880, advanced=True),
    Setting("history.max_chars", "历史总字符上限", 200000, maximum=200000, advanced=True),
    Setting("reference.mode", "默认材料覆盖模式", "full", kind="str", choices=("full", "relevant"),
            help="full 全文；relevant 仅关键词相关片段。"),
    Setting("reference.chunk_size", "材料分块字符数", 1600, advanced=True),
    Setting("reference.chunk_overlap", "材料分块重叠字符数", 160, minimum=0, advanced=True),
    Setting("reference.max_chunks", "相关材料片段数量上限", 6, advanced=True),
    Setting("reference.max_chars", "选中材料字符预算", optional=True, advanced=True),
    Setting("reference.max_files", "材料文件数量上限", 10, maximum=10, advanced=True),
    Setting("reference.max_file_bytes", "材料单文件字节上限", 5242880, maximum=5242880, advanced=True),
    Setting("reference.max_extracted_chars", "材料单文件提取字符上限", 200000, maximum=200000, advanced=True),
    Setting("reference.max_pdf_pages", "PDF 页数上限", 200, maximum=200, advanced=True),
]
_model_fields = [
    Setting("name", "模型名称", "", kind="str", suffix="NAME"),
    Setting("base_url", "API 基地址", "", kind="str", suffix="BASE_URL"),
    Setting("api_key", "API 密钥", "", kind="str", secret=True, suffix="API_KEY",
            help="隐藏输入；留空保留当前密钥。"),
    Setting("max_tokens", "普通生成／最终评审输出 token 上限", 8192, suffix="MAX_TOKENS"),
    Setting("context_window", "上下文窗口 token 上限", optional=True,
            help="填写服务实际限制；交接前必须配置。更换模型或地址后须重新确认窗口。"),
    Setting("temperature", "温度", kind="float", optional=True, minimum=0, maximum=2,
            advanced=True, suffix="TEMPERATURE", help="留空不发送温度参数。"),
    Setting("reasoning_effort", "推理强度", kind="str", optional=True,
            choices=tuple(sorted(REASONING_EFFORTS)), advanced=True,
            help="留空使用服务默认值；写入 EXTRA_BODY。"),
    Setting("timeout", "连接／发送／连接池超时（秒）", 90, kind="float", minimum=0.1,
            advanced=True, suffix="TIMEOUT", help="不限制响应读取等待。"),
    Setting("slow_warning_seconds", "无活动提醒（秒）", 90, kind="float", minimum=0.1,
            advanced=True, suffix="SLOW_WARNING_SECONDS", help="只提醒，不停止请求。"),
    Setting("token_limit_field", "输出限制字段", "max_tokens", kind="str", advanced=True,
            choices=("max_tokens", "max_completion_tokens"), suffix="TOKEN_LIMIT_FIELD"),
    Setting("json_mode", "JSON 模式", False, kind="bool", advanced=True, suffix="JSON_MODE"),
    Setting("extra_body", "额外请求参数 JSON", {}, kind="json", advanced=True, suffix="EXTRA_BODY",
            help="保留服务专有参数；不能覆盖连接、消息和输出限制字段。"),
]
SPECS = {spec.key: spec for spec in _runtime}
for _role in ROLE_PREFIXES:
    for _spec in _model_fields:
        _key = f"models.{_role}.{_spec.key}"
        _values = {**vars(_spec), "key": _key}
        if _role == "b" and _spec.key == "max_tokens":
            _values["help"] = "普通与交接流程的提示词生成共用此额度；B 不参与历史整理。"
        SPECS[_key] = Setting(**_values)
    if _role != "b":
        for _spec in (
            Setting("handoff_max_tokens", "交接输出 token 上限", optional=True,
                    suffix="HANDOFF_MAX_TOKENS", help="留空使用交接自动规则；下方显示实际额度。"),
            Setting("handoff_reasoning_effort", "交接推理强度", kind="str", optional=True, advanced=True,
                    choices=tuple(sorted(REASONING_EFFORTS)), suffix="HANDOFF_REASONING_EFFORT",
                    help="留空沿用普通模型配置。"),
        ):
            _key = f"models.{_role}.{_spec.key}"
            SPECS[_key] = Setting(**{**vars(_spec), "key": _key})


def default_settings():
    result = {"version": 1, "run": {}, "history": {}, "reference": {}}
    for spec in _runtime:
        group, key = spec.key.split(".")
        result[group][key] = deepcopy(spec.default)
    return result


def validate_settings(data):
    if (not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1
            or set(data) - {"version", "run", "history", "reference"}):
        raise ConfigurationError("运行参数文件版本或结构无效。")
    result = default_settings()
    for group in ("run", "history", "reference"):
        section = data.get(group, {})
        if not isinstance(section, dict) or set(section) - set(result[group]):
            raise ConfigurationError(f"运行参数 {group} 包含无效字段。")
        for key, value in section.items():
            result[group][key] = SPECS[f"{group}.{key}"].parse(value)
    if result["reference"]["chunk_overlap"] >= result["reference"]["chunk_size"]:
        raise ConfigurationError("材料重叠长度必须小于分块长度。")
    return result


def read_settings(root=ROOT):
    path = Path(root) / SETTINGS_NAME
    if not path.exists():
        return default_settings()
    try:
        with path.open("rb") as stream:
            raw = stream.read(1048577)
        if len(raw) > 1048576:
            raise ValueError
        return validate_settings(strict_json(raw.decode("utf-8-sig")))
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise ConfigurationError("运行参数文件无法读取或 JSON 无效；请通过配置程序修正。") from None


def run_options(settings, **overrides):
    from optimizer_engine import RunOptions
    values = deepcopy(settings["run"])
    values["retries"] = values.pop("network_max_attempts") - 1
    values.update(overrides)
    return RunOptions(**values)


def reference_options(settings, **overrides):
    from optimizer_documents import ReferenceOptions
    return ReferenceOptions(**(settings["reference"] | overrides))


def history_options(settings):
    from optimizer_handoff import HistoryOptions
    return HistoryOptions(**settings["history"])


def _json_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _replace_env_binding(binding, value):
    """Preserve spacing, export prefix and comments around a managed assignment."""
    encoded = "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    match = re.match(r"(\s*(?:export[ \t]+)?(?:'[^']*'|[^=\s#]+)[ \t]*=[ \t]*)(.*)",
                     binding.original.string, re.S)
    if match is None:
        return f"{binding.key}={encoded}\n"
    prefix, raw = match.groups()
    if raw.startswith(("'", '"')):
        quote, index = raw[0], 1
        while index < len(raw):
            if raw[index] == "\\":
                index += 2
            elif raw[index] == quote:
                return prefix + encoded + raw[index + 1:]
            else:
                index += 1
    comment = re.search(r"[ \t]+#", raw)
    ending = re.search(r"(?:\r\n|\r|\n)$", raw)
    suffix = raw[comment.start():] if comment else ending.group() if ending else ""
    return prefix + encoded + suffix


def _read_bytes(path):
    try:
        return path.read_bytes() if path.exists() else None
    except OSError:
        raise ConfigurationError("配置文件无法读取，请检查权限。") from None


def _guard_target(path, root):
    """Check aliases before reads/writes, including an explicitly chosen env path."""
    from optimizer_reports import _project_protected, _same_path
    original = Path(path).expanduser().absolute()
    for item in (original, *original.parents):
        if item.exists() or item.is_symlink():
            metadata = item.lstat()
            if (stat.S_ISLNK(metadata.st_mode)
                    or getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                raise ConfigurationError("配置路径不能经过符号链接或重解析点。")
    if original.exists() and (not original.is_file() or original.stat().st_nlink > 1):
        raise ConfigurationError("配置目标必须是普通文件，不能是目录或硬链接。")
    protected = [p for p in _project_protected(root)
                 if p.name not in {".env", "context_windows.json", SETTINGS_NAME}]
    protected += [Path(root) / ".env.example", Path(root) / "optimizer_settings.example.json",
                  Path(root) / "last_optimized_prompt.md"]
    if any(_same_path(original.resolve(), p.resolve()) for p in protected):
        raise ConfigurationError("配置保存不能覆盖源文件、示例或生成结果。")
    return original.resolve()


def _commit(files, originals, root):
    paths = [_guard_target(path, root) for path in files]
    if len(set(paths)) != len(paths):
        raise ConfigurationError("配置目标路径不能重合。")
    for path, original in originals.items():
        if _read_bytes(path) != original:
            raise ConfigurationError("配置已被其他程序修改，请重新打开后保存。")
    written = []
    try:
        for path, contents in files.items():
            atomic_write_bytes(path, contents)
            written.append(path)
    except OSError:
        failed = []
        for path in reversed(written):
            try:
                if originals[path] is None:
                    path.unlink(missing_ok=True)
                else:
                    atomic_write_bytes(path, originals[path])
            except OSError:
                failed.append(path.name)
        if failed:
            raise ConfigurationError("保存失败且以下文件未能回滚：" + "、".join(failed) + "；请重新检查配置。") from None
        raise ConfigurationError("配置保存失败，已回滚本次已写入文件。") from None


def prepare_window_config(configs, values, *, previous=None):
    """The one window validation/binding service, also used by the old CLI flag."""
    from optimizer_handoff import model_identity, validate_role_window
    previous = previous or {"version": 1, "roles": {}}
    result = {"version": 1, "roles": {}}
    for role in ("a", "b", "judge"):
        if role not in configs:
            if values.get(role) is not None:
                raise ConfigurationError(f"{role} 模型配置不完整，不能绑定窗口。")
            continue
        old = previous.get("roles", {}).get(role, {})
        identity = model_identity(configs[role])
        if role in values:
            value = values[role]
            if value is None:
                continue
            entry = {**identity, "context_window": value}
            validate_role_window(role, configs[role], entry)
            result["roles"][role] = entry
        elif isinstance(old, dict) and all(old.get(k) == v for k, v in identity.items()):
            validate_role_window(role, configs[role], old)
            result["roles"][role] = {**identity, "context_window": old["context_window"]}
    return result


class ConfigurationStore:
    """Private values stay inside this object and never enter public summaries."""
    def __init__(self, root=ROOT, *, env_path=UNSET, context_path=None, recover=False):
        self.root = Path(root).resolve()
        self.env_path = self.root / ".env" if env_path is UNSET else env_path
        self.env_path = _guard_target(self.env_path, self.root) if self.env_path is not None else None
        self.context_path = _guard_target(context_path or self.root / "context_windows.json", self.root)
        self.settings_path = _guard_target(self.root / SETTINGS_NAME, self.root)
        if self.env_path in {self.context_path, self.settings_path} or self.context_path == self.settings_path:
            raise ConfigurationError("模型、运行和窗口配置路径不能重合。")
        self.recovery_notes = []
        self._recovering_settings = False
        try:
            self.settings = read_settings(self.root)
        except ConfigurationError:
            if not recover:
                raise
            self.settings = default_settings()
            self._recovering_settings = True
            self.recovery_notes.append("运行参数文件无效，已在表单中恢复默认值；只有保存才会替换原文件，取消保留原文件。")
        self.originals = {p: _read_bytes(p) for p in (self.env_path, self.context_path, self.settings_path) if p is not None}
        try:
            saved = strict_json((self.originals[self.settings_path] or b'{}').decode("utf-8-sig"))
            self._saved_runtime_keys = {f"{group}.{name}" for group in ("run", "history", "reference")
                for name in saved.get(group, {})} if isinstance(saved, dict) and not self._recovering_settings else set()
        except (ValueError, UnicodeError, RecursionError, TypeError):
            self._saved_runtime_keys = set()
        try:
            self._env_text = (self.originals.get(self.env_path) or b"").decode("utf-8-sig")
        except UnicodeError:
            raise ConfigurationError("模型配置文件无法按 UTF-8 读取。") from None
        from dotenv import dotenv_values
        self._values = {k: v or "" for k, v in dotenv_values(
            stream=StringIO(self._env_text), interpolate=False).items()}
        try:
            self.windows = strict_json((self.originals[self.context_path] or b'{"version":1,"roles":{}}').decode("utf-8-sig"))
            if not isinstance(self.windows, dict) or not isinstance(self.windows.get("roles"), dict):
                raise ValueError
        except (ValueError, UnicodeError, RecursionError):
            self.windows = {"version": 1, "roles": {}}

    def _effective(self, values=None):
        return (self._values if values is None else values) | dict(os.environ)

    def _model_value(self, spec, values):
        _, role, key = spec.key.split(".")
        prefix = ROLE_PREFIXES[role]
        if spec.secret:
            return ""
        if key == "context_window":
            from optimizer_handoff import model_identity
            try:
                config = model_config(self._effective(values), prefix, INTERNAL_ROLES[role])
            except ConfigurationError:
                return None
            entry = self.windows.get("roles", {}).get(config.role, {})
            return entry.get("context_window") if isinstance(entry, dict) and all(
                entry.get(k) == v for k, v in model_identity(config).items()) else None
        if key == "reasoning_effort":
            try:
                extra = strict_json(values.get(prefix + "_EXTRA_BODY", "") or "{}")
                return extra.get("reasoning_effort") if isinstance(extra, dict) else None
            except (ValueError, RecursionError):
                return None
        raw = values.get(prefix + "_" + spec.suffix, "")
        if spec.optional and raw.strip() == "null":
            # .env uses an empty value, whereas CLI / JSON accept explicit null.
            raise ConfigurationError(f"{spec.key} 无效；模型文件中的可选值应留空。")
        return spec.parse(raw) if raw.strip() else deepcopy(spec.default)

    def form_values(self):
        values = self._values if self.env_path is not None else dict(os.environ)
        result = {}
        for key, spec in SPECS.items():
            if key.startswith("models."):
                try:
                    result[key] = self._model_value(spec, values)
                except ConfigurationError:
                    # An explicit form must be able to repair an invalid saved field.
                    role = key.split(".")[1]
                    result[key] = values.get(ROLE_PREFIXES[role] + "_" + spec.suffix, "")
            else:
                group, name = key.split(".")
                result[key] = deepcopy(self.settings[group][name])
        return result

    def public(self):
        result = deepcopy(self.settings)
        result["models"], result["sources"] = {}, {}
        effective = self._effective()
        for key, spec in SPECS.items():
            if not key.startswith("models."):
                result["sources"][key] = "参数文件" if key in self._saved_runtime_keys else "默认值"
                continue
            _, role, name = key.split(".")
            prefix = ROLE_PREFIXES[role]
            model = result["models"].setdefault(role, {})
            if name == "api_key":
                model["api_key_configured"] = bool(effective.get(prefix + "_API_KEY", "").strip())
                result["sources"][key] = ("进程环境覆盖" if prefix + "_API_KEY" in os.environ
                    else "模型文件" if self._values.get(prefix + "_API_KEY", "").strip() else "未配置")
                continue
            try:
                value = self._model_value(spec, effective)
            except ConfigurationError:
                value = "无效，请修正"
            if name == "extra_body":
                value = {"fields": sorted(value)} if isinstance(value, dict) else None
            elif name == "base_url" and value:
                try:
                    parsed = urlsplit(value)
                    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                            or parsed.username or parsed.password or parsed.query or parsed.fragment):
                        raise ValueError
                    _ = parsed.port
                except ValueError:
                    value = "无效，请修正"
            elif name == "reasoning_effort" and value is not None and (
                    not isinstance(value, str) or value not in REASONING_EFFORTS):
                value = "自定义值（内容隐藏，请检查服务支持情况）"
            model[name] = value
            source_key = prefix + "_" + (spec.suffix or "EXTRA_BODY")
            result["sources"][key] = ("窗口文件" if name == "context_window" else
                "进程环境覆盖" if source_key in os.environ else
                "模型文件" if self._values.get(source_key, "").strip() else "默认值")
        for role, prefix in ROLE_PREFIXES.items():
            try:
                config = model_config(effective, prefix, INTERNAL_ROLES[role])
                result["models"][role]["effective_handoff_max_tokens"] = handoff_model_config(config).max_tokens
            except ConfigurationError:
                pass
        return result

    def check(self):
        errors, models = [], {}
        for role, prefix in ROLE_PREFIXES.items():
            try:
                config = model_config(self._effective(), prefix, INTERNAL_ROLES[role])
                models[config.role] = config
            except ConfigurationError as error:
                errors.append(str(error))
        generation_ready = not errors
        from optimizer_handoff import validate_role_window
        if self.windows.get("version") != 1:
            errors.append("上下文窗口配置结构不正确，请重新填写。")
        else:
            for role, config in models.items():
                try:
                    validate_role_window(role, config, self.windows.get("roles", {}).get(role, {}))
                except ConfigurationError as error:
                    errors.append(str(error))
        return {"valid": not errors, "generation_ready": generation_ready,
                "handoff_ready": not errors, "errors": errors, "configuration": self.public()}

    def apply(self, patch, *, allow_secrets=False, dry_run=False):
        changes = {}
        for key, value in patch.items():
            key = key.replace("models.judge.", "models.c.")
            spec = SPECS.get(key)
            if spec is None:
                raise ConfigurationError("包含未知配置项，请用 --config-show 或 --help 查看。")
            if spec.secret and not allow_secrets:
                raise ConfigurationError("密钥不能通过普通命令参数传入，请使用 --configure 的隐藏输入。")
            if spec.secret and not value:
                continue
            parsed = spec.parse(value)
            if spec.secret and not parsed:
                continue
            changes[key] = parsed
        baseline = self.form_values()
        changes = {key: value for key, value in changes.items()
                   if SPECS[key].secret or key.endswith(".context_window") or value != baseline.get(key)}
        settings, values = deepcopy(self.settings), dict(self._values)
        env_updates, windows = {}, {}
        for key, value in changes.items():
            spec = SPECS[key]
            if not key.startswith("models."):
                group, name = key.split(".")
                settings[group][name] = value
                continue
            _, role, name = key.split(".")
            if name == "context_window":
                windows[INTERNAL_ROLES[role]] = value
            elif name != "reasoning_effort":
                if self.env_path is None:
                    raise ConfigurationError("--no-env 不能保存模型配置；请指定模型配置文件。")
                encoded = ("" if value is None else json.dumps(value, ensure_ascii=False, allow_nan=False)
                           if spec.kind in {"json", "bool"} else str(value))
                env_updates[ROLE_PREFIXES[role] + "_" + spec.suffix] = encoded
        values.update(env_updates)
        for key, value in changes.items():
            if key.endswith(".reasoning_effort"):
                if self.env_path is None:
                    raise ConfigurationError("--no-env 不能保存模型配置。")
                role = key.split(".")[1]
                env_key = ROLE_PREFIXES[role] + "_EXTRA_BODY"
                try:
                    extra = strict_json(values.get(env_key, "") or "{}")
                except (ValueError, RecursionError):
                    raise ConfigurationError("额外请求参数 JSON 无效。") from None
                if not isinstance(extra, dict):
                    raise ConfigurationError("额外请求参数必须是 JSON 对象。")
                if value is None:
                    extra.pop("reasoning_effort", None)
                else:
                    extra["reasoning_effort"] = value
                values[env_key] = env_updates[env_key] = json.dumps(extra, ensure_ascii=False, allow_nan=False)
        settings = validate_settings(settings)
        configs = {}
        effective = self._effective(values)
        for role, prefix in ROLE_PREFIXES.items():
            # Validate parameter fields even while initial required fields are incomplete.
            draft = dict(effective)
            complete = all(draft.get(prefix + "_" + name, "").strip() for name in ("NAME", "BASE_URL", "API_KEY"))
            for suffix, fallback in (("NAME", "configuration-placeholder"),
                                     ("BASE_URL", "http://127.0.0.1:9/v1"), ("API_KEY", "validation-placeholder")):
                if not draft.get(prefix + "_" + suffix, "").strip():
                    draft[prefix + "_" + suffix] = fallback
            config = model_config(draft, prefix, INTERNAL_ROLES[role])
            if complete:
                configs[config.role] = config
        window_data = prepare_window_config(configs, windows, previous=self.windows)
        files = {}
        if env_updates:
            from dotenv.parser import parse_stream
            lines, remaining = [], dict(env_updates)
            bindings = list(parse_stream(StringIO(self._env_text)))
            last = {binding.key: index for index, binding in enumerate(bindings) if binding.key in env_updates}
            for index, binding in enumerate(bindings):
                if binding.key in env_updates:
                    if index == last[binding.key]:
                        lines.append(_replace_env_binding(binding, remaining.pop(binding.key)))
                    else:
                        # Remove redundant definitions while keeping their comments.
                        comment_only = _replace_env_binding(binding, "")
                        if "#" in comment_only:
                            lines.append("#" + comment_only.split("#", 1)[1])
                else:
                    lines.append(binding.original.string)
            text = "".join(lines)
            if text and not text.endswith(("\r", "\n")):
                text += "\n"
            for key, value in remaining.items():
                value = value.replace("\\", "\\\\").replace("'", "\\'")
                text += f"{key}='{value}'\n"
            files[self.env_path] = text.encode("utf-8")
        if settings != self.settings or self._recovering_settings:
            files[self.settings_path] = _json_bytes(settings)
        if window_data != self.windows:
            files[self.context_path] = _json_bytes(window_data)
        warnings = []
        for role, prefix in ROLE_PREFIXES.items():
            if INTERNAL_ROLES[role] not in configs:
                warnings.append(f"模型 {role.upper()} 缺少名称、地址或密钥。")
            elif INTERNAL_ROLES[role] not in window_data["roles"]:
                warnings.append(f"模型 {role.upper()} 尚未绑定交接窗口。")
        if not dry_run:
            _commit(files, self.originals, self.root)
        changed = set(changes)
        if self._recovering_settings:
            changed.update(spec.key for spec in _runtime)
        return {"status": "configured", "changed": sorted(changed),
                "files": [str(p) for p in files], "applies_to": "new_session", "warnings": warnings}


def configure_windows(configs, values, *, path):
    """Compatibility writes use the same binder and transaction as every frontend."""
    root = Path(path).absolute().parent
    path = _guard_target(path, root)
    if set(values) != {"a", "b", "judge"}:
        raise ConfigurationError("请填写 A、B、C 的全部窗口。")
    parsed = {role: SPECS[f"models.{'c' if role == 'judge' else role}.context_window"].parse(value)
              for role, value in values.items()}
    if any(value is None for value in parsed.values()):
        raise ConfigurationError("交接窗口须为正整数。")
    result = prepare_window_config(configs, parsed)
    if set(result["roles"]) != {"a", "b", "judge"}:
        raise ConfigurationError("模型角色配置不完整。")
    _commit({path: _json_bytes(result)}, {path: _read_bytes(path)}, root)
    return result


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, (bool, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return str(value)


def configuration_cli(args, *, root, stdin, stdout, stderr):
    """No model construction or requests; keys are never command-line values."""
    from getpass import getpass
    store = ConfigurationStore(root, env_path=None if args.no_env else args.env_file or Path(root) / ".env",
                               context_path=args.context_config, recover=args.configure)
    if args.config_show:
        result, code = store.public(), 0
    elif args.config_check:
        result = store.check()
        code = 0 if result["valid"] else 2
    elif args.config_set:
        patch = {}
        for key, value in args.config_set:
            canonical = key.replace("models.judge.", "models.c.")
            if canonical in patch:
                raise ConfigurationError("同一配置项不能在一批变更中重复填写。")
            patch[canonical] = value
        result, code = store.apply(patch), 0
    else:
        if not getattr(stdin, "isatty", lambda: False)():
            raise ConfigurationError("--configure 需要交互终端；批量修改请使用 --config-set。")
        baseline, patch = store.form_values(), {}
        public = store.public()
        print("配置向导：回车保留；/clear 清空可选值；/cancel 取消。", file=stderr)
        for note in store.recovery_notes:
            print(note, file=stderr)
        print("包含高级设置？[y/N]", file=stderr, flush=True)
        answer = stdin.readline()
        if not answer or answer.strip() == "/cancel":
            raise KeyboardInterrupt
        advanced = answer.strip().lower() in {"y", "yes", "是"}
        for key, spec in SPECS.items():
            if spec.advanced and not advanced:
                continue
            shown = public["models"][key.split(".")[1]].get(key.split(".")[2]) if key.startswith("models.") else baseline[key]
            current = ("已配置／隐藏" if spec.secret else "JSON 内容隐藏" if spec.kind == "json"
                       else format_value(shown) or "留空")
            print(f"{key} · {spec.label} [{current}] {spec.help}", file=stderr, flush=True)
            if spec.secret:
                raw = getpass("密钥（留空保留）：", stream=stderr)
            else:
                raw = stdin.readline()
                if not raw:
                    raise KeyboardInterrupt
                raw = raw.rstrip("\r\n")
            if raw == "/cancel":
                raise KeyboardInterrupt
            if raw:
                patch[key] = None if raw == "/clear" else raw
        preview = store.apply(patch, allow_secrets=True, dry_run=True)
        print("将修改：" + "、".join(preview["changed"]), file=stderr)
        print("保存？[y/N]", file=stderr, flush=True)
        if stdin.readline().strip().lower() not in {"y", "yes", "是"}:
            result, code = {"status": "cancelled", "changed": []}, 130
        else:
            result, code = store.apply(patch, allow_secrets=True), 0
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), file=stdout)
    return code
