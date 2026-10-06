"""Validated outputs and a streaming Chat Completions adapter with activity signals."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import json
import os
import re
from typing import Annotated, Literal, Protocol

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator

from optimizer_config import ModelConfig, OptimizerError

def nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("Blank text")
    return value


Text = Annotated[str, Field(min_length=1, max_length=100000), AfterValidator(nonblank)]
Score = Annotated[int, Field(ge=0, le=5)]


class OutputError(OptimizerError):
    def __init__(self, message, *, usage=None):
        super().__init__(message)
        # Only sanitized numeric usage, never the provider's partial text/body.
        self.usage = usage


class ModelCallError(OptimizerError):
    pass


class TransientModelError(ModelCallError):
    pass


class StreamingUnsupportedError(ModelCallError):
    """An explicit parameter rejection, before any response content was received."""


@dataclass(frozen=True)
class ModelActivity:
    # No response text or reasoning text crosses this callback boundary.
    kind: Literal["reasoning", "output", "mode", "validating"]
    characters: int = 0
    mode: Literal["streaming", "non_streaming"] | None = None
    finish_reason: str | None = None


ActivityCallback = Callable[[ModelActivity], None]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Constraint(StrictModel):
    source_quote: Text
    constraint: Text


class Draft(StrictModel):
    status: Literal["ready", "needs_clarification"]
    optimized_prompt: Annotated[str, Field(min_length=1), AfterValidator(nonblank)]
    preserved_constraints: list[Constraint]
    clarification_questions: list[Text]
    change_summary: list[Text]

    @model_validator(mode="after")
    def consistent_status(self):
        if not self.optimized_prompt.strip():
            raise ValueError("Empty prompt")
        if bool(self.clarification_questions) != (self.status == "needs_clarification"):
            raise ValueError("Questions and status disagree")
        return self


class Finding(StrictModel):
    kind: Literal["goal_changed", "scope_expanded", "constraint_lost", "fact_added",
                  "format_conflict", "layer_boundary", "redundancy", "internal_conflict", "ineffective_steps", "uncertain"]
    source_quote: Text
    candidate_quote: str
    explanation: Text

    @model_validator(mode="after")
    def quality_evidence(self):
        if (self.kind in {"layer_boundary", "redundancy", "internal_conflict", "ineffective_steps"}
                and not self.candidate_quote.strip()):
            raise ValueError("Quality finding needs a candidate excerpt")
        return self


class CandidateReview(StrictModel):
    candidate_id: Text
    verdict: Literal["pass", "fail", "uncertain"]
    findings: list[Finding]
    clarity: Score
    conciseness: Score
    reason: Text

    @model_validator(mode="after")
    def consistent_verdict(self):
        if self.verdict == "pass" and self.findings:
            raise ValueError("Passed candidate has unresolved findings")
        if self.verdict != "pass" and not self.findings:
            raise ValueError("Failed/uncertain candidate needs evidence")
        return self


class Review(StrictModel):
    reviews: list[CandidateReview]
    action: Literal["select", "keep_original", "repair", "needs_clarification", "needs_review"]
    candidate_id: str | None
    clarification_questions: list[Text]
    reason: Text

    @model_validator(mode="after")
    def consistent_action(self):
        needs_id = self.action in {"select", "keep_original", "repair"}
        if needs_id != bool(self.candidate_id):
            raise ValueError("Decision and candidate ID disagree")
        if bool(self.clarification_questions) != (self.action == "needs_clarification"):
            raise ValueError("Decision and questions disagree")
        return self


def parse_output(text: str, schema: type[StrictModel]):
    # A single enclosing fence is accepted; prose, duplicate keys and coercions are not.
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines[0].lower() in {"```", "```json"} and lines[-1] == "```":
            text = "\n".join(lines[1:-1])

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    try:
        data = json.loads(text, object_pairs_hook=unique_keys)
        return schema.model_validate(data)
    except (ValueError, ValidationError, TypeError, RecursionError):
        # Pydantic errors can embed entire prompts or provider replies. Do not echo them.
        raise OutputError(f"模型输出不符合 {schema.__name__} JSON 结构。") from None


@dataclass
class ModelReply:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    returned_model: str | None = None


class ChatModel(Protocol):
    async def complete(self, system: str, payload: dict, *, timeout: float,
                       on_activity: ActivityCallback | None = None) -> ModelReply: ...

    def disable_streaming(self) -> None: ...

    async def close(self) -> None: ...


class LangChainChatModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = None
        self._sync_http = None
        self._async_http = None
        self.streaming = True

    def disable_streaming(self):
        self.streaming = False

    @staticmethod
    def _transport_timeout(seconds):
        from openai import Timeout
        # A slow model may stay silent indefinitely. Network setup remains bounded.
        return Timeout(seconds, read=None)

    def _client(self):
        if self.client is None:
            from langchain_openai import ChatOpenAI
            from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

            # Own the transports: LangChain caches its default clients across models.
            # Closing a cached client would break peers and later evaluation cases.
            network_timeout = self._transport_timeout(self.config.timeout)
            transport_options = {"timeout": network_timeout}
            if proxy := os.environ.get("OPENAI_PROXY"):
                transport_options["proxy"] = proxy
            self._sync_http = DefaultHttpxClient(**transport_options)
            self._async_http = DefaultAsyncHttpxClient(**transport_options)
            # Preserve provider-specific token field rather than allowing ChatOpenAI
            # to rename max_tokens to max_completion_tokens for every endpoint.
            extra = {**self.config.extra_body,
                     self.config.token_limit_field: self.config.max_tokens}
            # Explicit None prevents SDK defaults here; the SDK omits it on the wire.
            # Do not substitute zero for models with fixed or unsupported temperature.
            kwargs = dict(model=self.config.name, api_key=self.config.api_key,
                          base_url=self.config.base_url, temperature=self.config.temperature,
                          timeout=network_timeout, max_retries=0,
                          use_responses_api=False, extra_body=extra,
                          http_client=self._sync_http, http_async_client=self._async_http,
                          openai_proxy=None, http_socket_options=())
            if self.config.json_mode:
                kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
            self.client = ChatOpenAI(**kwargs)
        return self.client

    @staticmethod
    def _notify(callback, kind, characters=0, mode=None, finish_reason=None):
        if callback is not None:
            try:
                reason = finish_reason if finish_reason in {
                    "stop", "length", "max_tokens", "content_filter", "tool_calls", "function_call"} else None
                callback(ModelActivity(kind, characters, mode, reason))
            except Exception:
                # Display failures must not interrupt a paid model request.
                pass

    @staticmethod
    def _stream_rejected(error):
        if getattr(error, "status_code", None) not in {400, 422}:
            return False
        body = getattr(error, "body", None)
        if not isinstance(body, dict):
            return False
        detail = body.get("error", body)
        if not isinstance(detail, dict):
            return False
        parameter = detail.get("param") or detail.get("parameter")
        code = detail.get("code") or detail.get("type")
        if (isinstance(parameter, str) and parameter in {"stream", "stream_options", "stream_options.include_usage"}
                and isinstance(code, str) and code in {"unsupported_parameter", "unsupported_value", "not_supported",
                                                      "invalid_parameter", "invalid_request_error"}):
            return True
        # Compatible gateways may omit param/code. Only an explicit rejection
        # mentioning streaming qualifies; generic bad requests never fall back.
        message = detail.get("message")
        if not isinstance(message, str):
            return False
        target = r"\b(?:stream_options(?:\.include_usage)?|stream(?:ing)?)\b"
        quoted = rf"['\"]?{target}['\"]?"
        patterns = (
            rf"{quoted}\s*(?:is\s+)?(?:not\s+supported|unsupported|not\s+allowed)\b",
            rf"\b(?:does\s+not|doesn't)\s+support\s+{quoted}",
            rf"\b(?:unsupported|unrecognized|unknown)\s+(?:parameter|argument)\s*:?\s*{quoted}",
            rf"{quoted}\s+must\s+be\s+false\b",
            r"不支持(?:\s*流式|\s*stream(?:ing)?\b|\s*stream_options\b)",
        )
        return any(re.search(pattern, message, re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _usage(value):
        value = value if isinstance(value, dict) else {}
        result = {}
        for wire_key, key in (("prompt_tokens", "input_tokens"),
                              ("completion_tokens", "output_tokens"),
                              ("total_tokens", "total_tokens")):
            count = value.get(wire_key)
            result[key] = count if type(count) is int and count >= 0 else None
        return result

    def _reply(self, content, usage, model, finish_reason):
        reported_usage = self._usage(usage)
        if finish_reason in {"length", "max_tokens", "content_filter"}:
            raise OutputError(f"{self.config.role} 返回了被截断或未完成的响应。", usage=reported_usage)
        if finish_reason is None:
            raise TransientModelError(f"{self.config.role} 响应在完成前中断。")
        if isinstance(content, list):
            content = "\n".join(block.get("text", "") for block in content
                                if isinstance(block, dict) and block.get("type") == "text")
        if not isinstance(content, str) or not content.strip():
            raise OutputError(f"{self.config.role} 没有返回有效文本。", usage=reported_usage)
        return ModelReply(content, **reported_usage,
                          returned_model=model if isinstance(model, str) else None)

    async def complete(self, system: str, payload: dict, *, timeout: float,
                       on_activity: ActivityCallback | None = None) -> ModelReply:
        raw_response = None
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langsmith import tracing_context

            prompt = ChatPromptTemplate.from_messages([
                ("system", system.replace("{", "{{").replace("}", "}}")),
                ("human", "{payload}"),
            ])
            # No implicit cloud tracing, no tools, and no automatic SDK retries.
            with tracing_context(enabled=False):
                messages = await prompt.ainvoke(
                    {"payload": json.dumps(payload, ensure_ascii=False)}, config={"callbacks": []})
                client = self._client()
                wire = client._get_request_payload(messages, stream=self.streaming)
                if self.streaming:
                    wire["stream_options"] = {"include_usage": True}
                self._notify(on_activity, "mode", mode="streaming" if self.streaming else "non_streaming")
                raw_response = await client.async_client.with_raw_response.create(
                    **wire, timeout=self._transport_timeout(timeout))
                response = raw_response.http_response
                is_stream = "text/event-stream" in response.headers.get("content-type", "").lower()
                self._notify(on_activity, "mode", mode="streaming" if is_stream else "non_streaming")
                if not is_stream:
                    # Some compatible gateways ignore stream=True. Reuse their result;
                    # issuing another completion would duplicate work and accounting.
                    self.streaming = False
                    await response.aread()
                    value = response.json()
                    choice = value["choices"][0]
                    content = choice["message"].get("content")
                    if isinstance(content, str) and content:
                        self._notify(on_activity, "output", len(content))
                    self._notify(on_activity, "validating", finish_reason=choice.get("finish_reason"))
                    return self._reply(content, value.get("usage"), value.get("model"),
                                       choice.get("finish_reason"))

                parts, usage, returned_model, finish_reason = [], None, None, None
                async for chunk in raw_response.parse():
                    value = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                    if not isinstance(value, dict):
                        continue
                    if isinstance(value.get("usage"), dict):
                        usage = value["usage"]
                    if isinstance(value.get("model"), str):
                        returned_model = value["model"]
                    choices = value.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        self._notify(on_activity, "reasoning", len(reasoning))
                    # Never aggregate reasoning or retain it in reply/metadata.
                    reasoning = None
                    text = delta.get("content")
                    if isinstance(text, str) and text:
                        parts.append(text)
                        self._notify(on_activity, "output", len(text))
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                self._notify(on_activity, "validating", finish_reason=finish_reason)
                return self._reply("".join(parts), usage, returned_model, finish_reason)
        except OptimizerError:
            raise
        except Exception as error:
            if self.streaming and raw_response is None and self._stream_rejected(error):
                raise StreamingUnsupportedError(
                    f"{self.config.role} 接口明确不支持流式参数。") from None
            code = getattr(error, "status_code", None)
            transient = code in {408, 409, 429} or (isinstance(code, int) and code >= 500)
            transient |= type(error).__name__ in {
                "APITimeoutError", "APIConnectionError", "TimeoutError", "ReadError", "WriteError",
                "ConnectError", "RemoteProtocolError", "ReadTimeout", "WriteTimeout", "ConnectTimeout", "PoolTimeout"}
            error_type = TransientModelError if transient else ModelCallError
            raise error_type(f"{self.config.role} 模型请求失败；请核对接口、模型参数或服务状态。") from None
        finally:
            if raw_response is not None:
                await raw_response.http_response.aclose()

    async def close(self):
        try:
            if self._async_http is not None:
                await self._async_http.aclose()
        finally:
            if self._sync_http is not None:
                self._sync_http.close()
            self._sync_http = None
            self._async_http = None
            self.client = None
