"""Explicit configuration loading; importing this module does not read .env."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent


class OptimizerError(Exception):
    """Messages from this exception are safe to display without provider details."""


class ConfigurationError(OptimizerError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    role: str
    name: str
    base_url: str
    api_key: str = field(repr=False)
    temperature: float | None = None
    timeout: float = 90.0
    max_tokens: int = 8192
    token_limit_field: str = "max_tokens"
    json_mode: bool = False
    extra_body: dict = field(default_factory=dict, repr=False)
    slow_warning_seconds: float = 90.0


def read_environment(env_file: Path | None = ROOT / ".env") -> dict[str, str]:
    values: dict[str, str] = {}
    if env_file is not None:
        if not env_file.is_file():
            raise ConfigurationError("找不到配置文件；可用 --no-env 仅使用进程环境变量。")
        try:
            from dotenv import dotenv_values

            # Do not execute shell syntax or silently expand secret references.
            values.update({k: v or "" for k, v in dotenv_values(
                env_file, encoding="utf-8-sig", interpolate=False
            ).items()})
        except (OSError, UnicodeError):
            raise ConfigurationError("配置文件无法按 UTF-8 读取。") from None
    # Explicit process settings take precedence, including an explicitly empty value.
    values.update(os.environ)
    return values


def _number(values: Mapping[str, str], key: str, default: str, *, integer=False,
            minimum=0, maximum=float("inf")) -> float | int:
    raw = values.get(key, "").strip() or default
    try:
        result = int(raw) if integer else float(raw)
        if not math.isfinite(result) or not minimum <= result <= maximum:
            raise ValueError
        return result
    except (ValueError, OverflowError):
        raise ConfigurationError(f"{key} 不是有效的范围内数值。") from None


def model_config(values: Mapping[str, str], prefix: str, role: str) -> ModelConfig:
    def get(suffix: str) -> str:
        return values.get(f"{prefix}_{suffix}", "").strip()

    missing = [f"{prefix}_{suffix}" for suffix in ("NAME", "API_KEY", "BASE_URL")
               if not get(suffix)]
    if missing:
        raise ConfigurationError("缺少配置项：" + ", ".join(missing))
    base_url = get("BASE_URL").rstrip("/")
    try:
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError
        _ = parsed.port
    except ValueError:
        raise ConfigurationError(f"{prefix}_BASE_URL 必须是不含凭据或查询参数的 HTTP(S) API 基地址。") from None

    temperature = None
    if get("TEMPERATURE"):
        temperature = _number(values, f"{prefix}_TEMPERATURE", "0", maximum=2)
    def reject_constant(value):
        raise ValueError("Non-finite JSON value")

    try:
        extra = json.loads(get("EXTRA_BODY") or "{}", parse_constant=reject_constant)
    except (ValueError, RecursionError):
        raise ConfigurationError(f"{prefix}_EXTRA_BODY 必须是 JSON 对象。") from None
    reserved = {"model", "messages", "stream", "stream_options", "n", "max_tokens", "max_completion_tokens",
                "temperature", "response_format", "api_key", "timeout"}
    if not isinstance(extra, dict) or reserved.intersection(extra):
        raise ConfigurationError(f"{prefix}_EXTRA_BODY 不能覆盖模型、消息、数量、长度或连接参数。")
    token_field = get("TOKEN_LIMIT_FIELD") or "max_tokens"
    if token_field not in {"max_tokens", "max_completion_tokens"}:
        raise ConfigurationError(f"{prefix}_TOKEN_LIMIT_FIELD 只能是 max_tokens 或 max_completion_tokens。")
    json_mode = get("JSON_MODE").lower() or "false"
    if json_mode not in {"true", "false"}:
        raise ConfigurationError(f"{prefix}_JSON_MODE 只能是 true 或 false。")
    return ModelConfig(
        role=role, name=get("NAME"), base_url=base_url, api_key=get("API_KEY"),
        temperature=temperature,
        timeout=_number(values, f"{prefix}_TIMEOUT", "90", minimum=0.1, maximum=3600),
        max_tokens=_number(values, f"{prefix}_MAX_TOKENS", "8192", integer=True,
                           minimum=128, maximum=131072),
        token_limit_field=token_field, json_mode=json_mode == "true", extra_body=extra,
        slow_warning_seconds=_number(values, f"{prefix}_SLOW_WARNING_SECONDS", "90", minimum=0.1),
    )


def load_models(values: Mapping[str, str]) -> dict[str, ModelConfig]:
    """Load the sole product pipeline; development baselines load A explicitly."""
    result = {}
    errors = []
    for role, prefix in (("a", "GENERATOR_A"), ("b", "GENERATOR_B"), ("judge", "JUDGE")):
        try:
            result[role] = model_config(values, prefix, role)
        except ConfigurationError as error:
            errors.append(str(error))
    if errors:
        raise ConfigurationError("\n".join(errors))
    return result
