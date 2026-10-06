"""Generation and independent review, with an injectable model adapter."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import inspect
from pathlib import Path
import random
import time

from optimizer_config import ConfigurationError, ModelConfig, OptimizerError
from optimizer_models import (
    Draft, LangChainChatModel, ModelActivity, ModelCallError, ModelReply, OutputError,
    StreamingUnsupportedError, TransientModelError, parse_output,
)
from optimizer_prompts import (
    DIALOGUE_POLICY, PROMPT_VERSION, clarification_prompt, prepare_generation_prompt, review_prompt,
)
from optimizer_dialogue import (
    UNSET, ClarificationDecision, DialogueQuestion, DialogueQuestionBatch, DialogueRequest,
    DialogueSession, fingerprint_request, normalize_response,
)
from optimizer_selection import DEFAULT_MAX_BUILTIN_EXAMPLES, DEFAULT_MAX_EXAMPLE_CHARS, validate_example_limits
from optimizer_review import evaluate_review, require_review_backend
from optimizer_layers import (
    Layer, LayerAnalysis, adds_reference_ban_after_omission, analysis_prompt,
    changed_model_reference_block, validate_decisions,
)
from optimizer_handoff import (
    FidelityFailed, HandoffError, HistoryBundle, HistoryPipeline,
    WindowExceeded, WindowLimits, read_window_config,
    history_fingerprint,
)


class BudgetExceeded(OptimizerError):
    pass


@dataclass(frozen=True)
class RunOptions:
    allow_repair: bool = True
    retries: int = 1
    token_budget: int | None = None
    max_input_chars: int | None = None
    choose_layers: bool = False
    max_builtin_examples: int = DEFAULT_MAX_BUILTIN_EXAMPLES
    max_example_chars: int = DEFAULT_MAX_EXAMPLE_CHARS

    def __post_init__(self):
        validate_example_limits(self.max_builtin_examples, self.max_example_chars)
        if (self.max_input_chars is not None and (
                    type(self.max_input_chars) is not int or self.max_input_chars < 1)
                or type(self.retries) is not int or not 0 <= self.retries <= 5):
            raise ConfigurationError("输入长度须为正整数或不设上限；重试次数须为 0 到 5。")
        if self.token_budget is not None and (type(self.token_budget) is not int or self.token_budget < 1):
            raise ConfigurationError("token 预算必须是正整数。")


@dataclass
class Candidate:
    id: str
    origin: str
    draft: Draft

    def anonymous(self):
        # The judge never sees the generating model or its claimed constraint coverage.
        return {"candidate_id": self.id, "status": self.draft.status,
                "text": self.draft.optimized_prompt,
                "clarification_questions": self.draft.clarification_questions}

    def to_dict(self):
        return {"id": self.id, "origin": self.origin, **self.draft.model_dump()}


@dataclass
class OptimizationResult:
    status: str
    optimized_prompt: str | None
    reviewed: bool
    questions: list[str] = field(default_factory=list)
    reason: str = ""
    selected_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    reviews: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    layers: list[dict] = field(default_factory=list)
    layer_decisions: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class Optimizer:
    # Development baselines override these hooks without adding a product mode.
    _required_model_roles = ("a", "b", "judge")
    _requires_independent_review = True

    def __init__(self, configs: dict[str, ModelConfig], options: RunOptions,
                 *, factory=LangChainChatModel, rng=None, layer_resolver=None, references=None,
                 history=None, context_windows=None, progress=None):
        self.configs = configs
        self.options = options
        self.workflow = "normal"
        self.factory = factory
        self.rng = rng if rng is not None else random.SystemRandom()
        self.models = {}
        self.calls = []
        self.review_history = []
        self.review_backend = None
        self.warnings = []
        self.started = None
        self.original = ""
        self.known_tokens = 0
        self.unknown_usage = 0
        self.layer_resolver = layer_resolver
        self.layer_analysis = None
        self.layer_decisions = []
        self.user_wait_seconds = 0.0
        self.example_selections = []
        self.references = references
        self.history = history
        self.context_windows = context_windows
        self.progress = progress
        self.windows = None
        self.history_pipeline = None
        self.handoff_context = None
        self.run_status = "pending"
        self.dialogue_session = None
        self.dialogue_event = None
        self._api_mode = None
        self._dialogue_busy = False
        self._dialogue_options = None
        self._dialogue_configs = None
        self._closed = False
        self._dialogue_candidates = []
        self._dialogue_handoff_state = None
        self._dialogue_handoff_identity = None
        self._dialogue_last_handoff_pipeline = None
        self._active_calls = {}
        self._activity_clock = time.monotonic
        self._streaming_fallback_roles = set()

    def _activate_workflow(self, workflow):
        self.workflow = workflow

    def _limits(self):
        return asdict(self.options)

    def _prepare_workflow(self, history, context_windows):
        """Validate detached inputs before changing a live session or spending calls."""
        if history is None:
            if context_windows is not None:
                raise ConfigurationError("没有历史时不能绑定交接窗口。")
            return "normal", None, None
        if (not isinstance(history, HistoryBundle) or not history.sources
                or any(not isinstance(source.document.page_content, str)
                       or not source.document.page_content.strip() or "\x00" in source.document.page_content
                       for source in history.sources)):
            raise ConfigurationError("历史输入无效，请重新加载非空的 TXT / Markdown 或粘贴旧记录。")
        data = context_windows
        if isinstance(data, (str, Path)):
            data = read_window_config(Path(data))
        windows = WindowLimits.from_config(data, self.configs)
        return "handoff", copy.deepcopy(data), windows

    def _clear_active_handoff(self):
        self.handoff_context = None
        self.history_pipeline = None
        self.windows = None
        self._dialogue_handoff_state = None
        self._dialogue_handoff_identity = None
        self._dialogue_last_handoff_pipeline = None
        if self.dialogue_session is not None:
            state = self.dialogue_session
            # Ordinary clarification can also have been asked using the retired
            # history. Re-evaluate all unresolved questions against the new input.
            state.pending_questions.clear()
            state.pending_delta.clear()

    def _elapsed(self):
        if self.started is None:
            return 0.0
        now = (self.dialogue_session.paused_at if self.dialogue_session is not None
               and self.dialogue_session.paused_at is not None else time.monotonic())
        return max(0.0, now - self.started)

    def _reserve(self, role, purpose, attempt=1):
        # No await here: parallel generators check token usage atomically.
        if self.options.token_budget is not None:
            if self.unknown_usage:
                raise BudgetExceeded("存在未知 token 用量，已停止后续请求。")
            if self.known_tokens >= self.options.token_budget:
                raise BudgetExceeded("已达到 token 调度预算，已停止后续请求。")
        record = {"role": role, "purpose": purpose, "model": self.configs[role].name,
                  "status": "running", "input_tokens": None, "output_tokens": None,
                  "total_tokens": None, "elapsed_seconds": 0.0,
                  "request_id": len(self.calls) + 1, "attempt": attempt,
                  "call_mode": "non_streaming" if role in self._streaming_fallback_roles else "streaming",
                  "activity_state": "waiting", "first_activity_seconds": None,
                  "last_activity_seconds": None, "reasoning_chars": 0, "output_chars": 0,
                  "slow_waiting": False, "finish_reason": None}
        self.calls.append(record)
        state = self.dialogue_session
        record.update(session_id=state.session_id if state else None, revision=state.revision if state else None)
        started = self._activity_clock()
        self._active_calls[record["request_id"]] = {
            "record": record, "started": started, "last_activity": started,
            "last_emit": started, "session_id": state.session_id if state else None,
            "revision": state.revision if state else None}
        self._emit_request_event("stage", record)
        self._emit_activity(record)
        return record

    def activity_snapshot(self):
        """Small, detached UI snapshot; no prompts, histories, or response text."""
        now = self._activity_clock()
        active = []
        state = self.dialogue_session
        context = (state.session_id, state.revision) if state else (None, None)
        for entry in self._active_calls.values():
            if context != (entry["session_id"], entry["revision"]):
                continue
            record = entry["record"]
            idle = max(0.0, now - entry["last_activity"])
            if record["status"] == "running":
                record["elapsed_seconds"] = round(max(0.0, now - entry["started"]), 3)
                record["slow_waiting"] = (record["activity_state"] != "validating"
                                           and idle >= self.configs[record["role"]].slow_warning_seconds)
            active.append({**{key: record[key] for key in (
                "request_id", "attempt", "role", "purpose", "model", "call_mode",
                "activity_state", "reasoning_chars", "output_chars", "elapsed_seconds",
                "first_activity_seconds", "last_activity_seconds", "slow_waiting")},
                "idle_seconds": round(idle, 3), "session_id": entry["session_id"],
                "revision": entry["revision"]})
        state = self.dialogue_session
        return {"active_calls": active, "session_id": state.session_id if state else None,
                "revision": state.revision if state else None, "workflow": self.workflow,
                "request_count": len(self.calls), "known_total_tokens": self.known_tokens,
                "unknown_usage_requests": self.unknown_usage,
                "total_tokens": None if self.unknown_usage else self.known_tokens,
                "elapsed_seconds": round(self._elapsed(), 3)}

    def _emit_request_event(self, kind, record):
        state = self.dialogue_session
        context = (state.session_id, state.revision) if state else (None, None)
        if context != (record.get("session_id"), record.get("revision")):
            return
        self._emit_dialogue(kind, phase=record["purpose"], role=record["role"],
                           request_id=record["request_id"], **self.activity_snapshot())

    def _emit_activity(self, record):
        self._emit_request_event("activity", record)

    def _on_model_activity(self, record, activity):
        entry = self._active_calls.get(record["request_id"])
        state = self.dialogue_session
        context = (state.session_id, state.revision) if state else (None, None)
        if (entry is None or record["status"] != "running" or self._closed
                or context != (entry["session_id"], entry["revision"])
                or not isinstance(activity, ModelActivity)):
            return
        before = (record["activity_state"], record["slow_waiting"], record["call_mode"])
        now = self._activity_clock()
        if activity.kind == "mode" and activity.mode in {"streaming", "non_streaming"}:
            record["call_mode"] = activity.mode
        elif activity.kind == "validating":
            record.update(activity_state="validating", slow_waiting=False)
            if activity.finish_reason in {"stop", "length", "max_tokens", "content_filter", "tool_calls", "function_call"}:
                record["finish_reason"] = activity.finish_reason
        elif (activity.kind in {"reasoning", "output"}
              and type(activity.characters) is int and activity.characters > 0):
            elapsed = round(max(0.0, now - entry["started"]), 3)
            if record["first_activity_seconds"] is None:
                record["first_activity_seconds"] = elapsed
            record["last_activity_seconds"] = elapsed
            record["reasoning_chars" if activity.kind == "reasoning" else "output_chars"] += activity.characters
            record.update(activity_state="thinking" if activity.kind == "reasoning" else "receiving",
                          slow_waiting=False)
            entry["last_activity"] = now
        else:
            return
        after = (record["activity_state"], record["slow_waiting"], record["call_mode"])
        if before != after or now - entry["last_emit"] >= 1:
            entry["last_emit"] = now
            self._emit_activity(record)

    async def _watch_activity(self, record):
        while True:
            await asyncio.sleep(1)
            if record["request_id"] not in self._active_calls:
                return
            self._emit_activity(record)

    async def _call(self, role, system, payload, purpose):
        estimate = self.windows.check(role, self.configs[role], system, payload) if self.windows else None
        attempt, retries_used = 0, 0
        while True:
            attempt += 1
            record = self._reserve(role, purpose, attempt)
            if estimate is not None:
                record["window_check"] = estimate
            started = self._active_calls[record["request_id"]]["started"]
            watcher = asyncio.create_task(self._watch_activity(record))
            retry = False
            fallback = False
            try:
                if role not in self.models:
                    self.models[role] = self.factory(self.configs[role])
                timeout = self.configs[role].timeout
                reply: ModelReply = await self.models[role].complete(
                    system, payload, timeout=timeout,
                    on_activity=lambda activity, request_record=record: self._on_model_activity(request_record, activity))
                record.update(status="ok", input_tokens=reply.input_tokens,
                              output_tokens=reply.output_tokens, total_tokens=reply.total_tokens,
                              returned_model=reply.returned_model)
                if reply.total_tokens is None:
                    self.unknown_usage += 1
                else:
                    self.known_tokens += reply.total_tokens
                return reply
            except asyncio.CancelledError:
                record["status"] = "cancelled"
                self.unknown_usage += 1
                raise
            except StreamingUnsupportedError:
                record["status"] = "stream_unsupported"
                self.unknown_usage += 1
                if self.options.token_budget is not None:
                    raise BudgetExceeded("流式请求失败后的用量未知；为遵守 token 调度预算已停止。") from None
                if role in self._streaming_fallback_roles:
                    raise ModelCallError(f"{role} 接口不支持当前请求参数。") from None
                self._streaming_fallback_roles.add(role)
                self.models[role].disable_streaming()
                retry, fallback = True, True
            except (TransientModelError, TimeoutError):
                record["status"] = "transient_error"
                self.unknown_usage += 1
                if self.options.token_budget is not None:
                    raise BudgetExceeded("请求失败后的用量未知；为遵守 token 调度预算已停止。") from None
                if retries_used >= self.options.retries:
                    raise ModelCallError(f"{role} 请求超时或暂时不可用，重试已用尽。") from None
                retries_used += 1
                retry = True
            except OptimizerError as error:
                record["status"] = "error"
                usage = getattr(error, "usage", None)
                if isinstance(usage, dict):
                    for key in ("input_tokens", "output_tokens", "total_tokens"):
                        value = usage.get(key)
                        record[key] = value if type(value) is int and value >= 0 else None
                if record["total_tokens"] is None:
                    self.unknown_usage += 1
                else:
                    self.known_tokens += record["total_tokens"]
                raise
            except Exception:
                record["status"] = "error"
                self.unknown_usage += 1
                raise ModelCallError(f"{role} 请求失败；服务响应详情未写入日志。") from None
            finally:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
                record["elapsed_seconds"] = round(max(0.0, self._activity_clock() - started), 3)
                record["slow_waiting"] = False
                record["activity_state"] = ("completed" if record["status"] == "ok" else
                                            "cancelled" if record["status"] == "cancelled" else "failed")
                if not retry:
                    self._active_calls.pop(record["request_id"], None)
                self._emit_request_event("usage", record)
                if not retry:
                    self._emit_activity(record)
            if retry:
                record["activity_state"] = "retrying"
                self._emit_activity(record)
                try:
                    delay = 0 if fallback else min(0.5 * (2 ** (retries_used - 1)), 4.0)
                    await asyncio.sleep(delay)
                finally:
                    self._active_calls.pop(record["request_id"], None)
                    record["activity_state"] = "failed"

    def _request_payload(self, *, analysis=False):
        payload = {"original_request": self.original}
        if self.options.choose_layers:
            payload["layer_decisions"] = [d.model_dump() for d in self.layer_decisions]
        if self.references is not None and self.references.files:
            payload["reference_files"] = self.references.payload(analysis=analysis)
        if self.handoff_context is not None:
            payload["handoff_context"] = self.handoff_context.prompt_payload()
        if self.dialogue_session is not None:
            payload["dialogue_request"] = {"session_id": self.dialogue_session.session_id,
                "revision": self.dialogue_session.revision, "fingerprint": self.dialogue_session.fingerprint}
        return payload

    def _evidence_texts(self):
        return ([self.original] + [d.value for d in self.layer_decisions if d.choice != "omit"]
                + (self.references.blocks if self.references is not None else [])
                + (self.handoff_context.blocks if self.handoff_context is not None else []))

    async def _prepare_layers(self):
        role = "a"
        system = analysis_prompt()
        # One schema correction, using the same token budget. Never
        # proceed to user choices with advice masquerading as reference material.
        for attempt in range(2):
            reply = await self._call(role, system,
                {**self._request_payload(analysis=True), "phase": "layer_analysis"},
                "layer_analysis_repair" if attempt else "layer_analysis")
            try:
                analysis = parse_output(reply.text, LayerAnalysis)
                break
            except OutputError:
                if attempt:
                    raise OutputError("缺层识别连续两次未符合结构要求，已停止生成。") from None
                self.warnings.append("缺层识别未符合结构要求，已请求一次重新识别。")
                system += ("\n上次识别不符合结构要求。请重新返回完整八层 JSON，核对字段、状态及引用；"
                           "缺失的 references 必须包含具体 reference_materials，每份有类型、用途和遵循方式；"
                           "按任务选择实现、交付物、执行案例或输入/输出示例，不能以建议菜单或禁止参考代替。")
        if self.references is not None and self.references.files:
            # Attached files supply this layer deterministically. A proposed
            # missing-reference menu cannot let later omission drop an attachment.
            analysis.layers = [Layer(layer="references", status="present",
                                     source_quotes=[f.evidence_quote for f in self.references.files],
                                     question="", suggestion="", needs_confirmation=False)
                               if layer.layer == "references" else layer for layer in analysis.layers]
        if self.handoff_context is not None:
            # The populated snapshot is authoritative for layer presence; do not
            # ask the user to provide the same context or references again.
            populated = {"context": self.handoff_context.context_block}
            if self.handoff_context.reference_blocks:
                populated["references"] = self.handoff_context.reference_blocks[0]
            analysis.layers = [Layer(layer=layer.layer, status="present", source_quotes=[populated[layer.layer]],
                                     question="", suggestion="", needs_confirmation=False)
                               if layer.layer in populated else layer for layer in analysis.layers]
        layer_evidence = [self.original] + (self.handoff_context.blocks if self.handoff_context else [])
        if any(not any(quote in t for t in layer_evidence) for layer in analysis.layers if layer.layer != "references"
               for quote in layer.source_quotes) or (
                not (self.references is not None and self.references.files)
                and any(not any(quote in t for t in layer_evidence) for layer in analysis.layers if layer.layer == "references"
                        for quote in layer.source_quotes)):
            raise OutputError("层次识别引用了原文中不存在的内容。")
        self.layer_analysis = analysis
        if not analysis.missing():
            return None
        if self.layer_resolver is None:
            return self._finish("needs_clarification", [], questions=[i.question for i in analysis.missing()],
                                reason="缺失层尚未选择用户增加、模型补全或省略。")
        waiting = time.monotonic()
        try:
            values = self.layer_resolver(analysis)
        except KeyboardInterrupt:
            values = None
        finally:
            elapsed = time.monotonic() - waiting
            self.user_wait_seconds += elapsed
            self.started += elapsed
        if values is None:
            return self._finish("cancelled", [], reason="已取消缺失层选择，未开始生成。")
        self.layer_decisions = validate_decisions(analysis, values)
        if (self.options.max_input_chars is not None
                and sum(len(t) for t in self._evidence_texts()) > self.options.max_input_chars):
            raise ConfigurationError("原始需求与补充内容合计超过本次允许的字符数。")
        return None

    async def _generate(self, role, repair=None):
        payload = self._request_payload()
        if repair is not None:
            payload["repair"] = repair
        adopted_model_reference = any(d.layer == "references" and d.choice == "model"
                                      for d in self.layer_decisions)
        kinds = {material.kind for layer in self.layer_analysis.layers if layer.layer == "references"
                 for material in layer.reference_materials} \
            if self.layer_analysis is not None and adopted_model_reference else set()
        if self.references is not None:
            kinds.update(file.kind for file in self.references.files)
        system, selection = prepare_generation_prompt(
            "repair" if repair else role, self.original, layer_decisions=self.layer_decisions,
            reference_kinds=kinds, max_builtin_examples=self.options.max_builtin_examples,
            adopted_file_reference=bool(self.references is not None and self.references.files),
            max_example_chars=self.options.max_example_chars,
        )
        self.example_selections.append({"role": role, "purpose": "repair" if repair else "generate", **selection})
        if self.dialogue_session is not None:
            system += "\n\n" + DIALOGUE_POLICY
        reply = await self._call(role, system,
                                 payload, "repair" if repair else "generate")
        draft = parse_output(reply.text, Draft)
        if any(not c.source_quote.strip() or not any(c.source_quote in t for t in self._evidence_texts())
               for c in draft.preserved_constraints):
            raise OutputError("生成结果引用了原文、已选补充或文件参考中不存在的约束。")
        if self.references is not None and draft.optimized_prompt:
            draft.optimized_prompt = self.references.attach(draft.optimized_prompt)
        if self.handoff_context is not None:
            draft.optimized_prompt = self.handoff_context.attach(draft.optimized_prompt, self.original)
            if self.options.max_input_chars is not None and len(draft.optimized_prompt) > self.options.max_input_chars:
                raise WindowExceeded("完整交接提示词超出正式输入字符上限；未裁剪关键内容。")
            self.windows.check(role, self.configs[role], "", {"prompt": draft.optimized_prompt})
        return draft

    @staticmethod
    def _validate_review(review, original, candidates, supplemental_texts=()):
        by_id = {c.id: c for c in candidates}
        ids = [r.candidate_id for r in review.reviews]
        if len(ids) != len(set(ids)) or set(ids) != set(by_id):
            raise OutputError("评审没有完整且唯一地覆盖候选。")
        for result in review.reviews:
            candidate = by_id[result.candidate_id]
            for finding in result.findings:
                if (not finding.source_quote.strip()
                        or not any(finding.source_quote in t for t in (original, *supplemental_texts))
                        or (finding.candidate_quote
                            and finding.candidate_quote not in candidate.draft.optimized_prompt)):
                    raise OutputError("评审证据无法在原文或对应候选中定位。")
        if review.candidate_id is not None:
            if review.candidate_id not in by_id:
                raise OutputError("评审选择了不存在的候选。")
            chosen = by_id[review.candidate_id]
            assessment = next(r for r in review.reviews if r.candidate_id == chosen.id)
            if review.action in {"select", "keep_original"}:
                if assessment.verdict != "pass" or chosen.draft.status != "ready":
                    raise OutputError("评审试图采用未通过或仍待确认的候选。")
                if (review.action == "keep_original") != (chosen.origin == "original"):
                    raise OutputError("评审动作与所选候选类型不一致。")
                if chosen.origin == "original" and any(text not in chosen.draft.optimized_prompt
                                                       for text in supplemental_texts):
                    raise OutputError("原文未包含已选补充，不能直接作为最终结果。")
            if review.action == "repair":
                if (chosen.origin == "original" or assessment.verdict != "fail"
                        or any(f.kind == "uncertain" for f in assessment.findings)):
                    raise OutputError("该评审意见不符合自动修复条件。")
                # The judge must choose a passed alternative before spending a
                # repair request. An original missing supplements is not usable.
                if any(result.verdict == "pass" and by_id[result.candidate_id].draft.status == "ready"
                       and (by_id[result.candidate_id].origin != "original"
                            or all(text in by_id[result.candidate_id].draft.optimized_prompt
                                   for text in supplemental_texts))
                       for result in review.reviews):
                    raise OutputError("已有可采用的合格候选，不能发起自动修复。")

    async def _review(self, candidates, purpose="review"):
        payload = {**self._request_payload(),
                   "original_candidate_id": next(c.id for c in candidates if c.origin == "original"),
                   "candidates": [c.anonymous() for c in candidates]}
        async def request(system, material):
            return await self._call("judge", system, material, purpose)

        system = review_prompt() + ("\n\n" + DIALOGUE_POLICY if self.dialogue_session else "")
        review = await evaluate_review(system, payload, request)
        self._validate_review(review, self.original, candidates, self._evidence_texts()[1:])
        self.review_history.append({"purpose": purpose,
                                    "order": [c.id for c in candidates], **review.model_dump(),
                                    **({"revision": self.dialogue_session.revision}
                                       if self.dialogue_session else {})})
        return review

    def metadata(self):
        activity = self.activity_snapshot()
        return {"mode": "quality", "workflow": self.workflow, "prompt_version": PROMPT_VERSION,
                "review_backend": self.review_backend,
                "input_sha256": hashlib.sha256(self.original.encode("utf-8")).hexdigest(),
                "elapsed_seconds": round(self._elapsed(), 3),
                "user_wait_seconds": round(self.user_wait_seconds, 3),
                "request_count": len(self.calls), "known_total_tokens": self.known_tokens,
                "unknown_usage_requests": self.unknown_usage,
                "total_tokens": None if self.unknown_usage else self.known_tokens,
                "calls": copy.deepcopy(self.calls), "limits": self._limits(),
                "active_calls": activity["active_calls"],
                "example_selections": list(self.example_selections),
                "reference_files": self.references.metadata() if self.references is not None else None,
                **({"context_windows": self.windows.metadata() if self.windows else None,
                    "handoff": {**(self.history_pipeline.metadata() if self.history_pipeline else {
                        "status": "not_started", "sources": [s.summary() for s in self.history.sources] if self.history else [],
                        "chunks": [], "stages": [], "snapshot": None, "raw_history_saved": False}),
                        "run_status": self.run_status,
                        "usage": [dict(c) for c in self.calls if c["purpose"].startswith(("extract-", "merge-", "update-"))]}}
                   if self.workflow == "handoff" else {}),
                "models": {role: {"name": c.name, "temperature": c.temperature,
                                   "max_tokens": c.max_tokens, "timeout": c.timeout,
                                   "slow_warning_seconds": c.slow_warning_seconds,
                                   "token_limit_field": c.token_limit_field,
                                   "json_mode": c.json_mode,
                                   "extra_body_fields": sorted(c.extra_body)}
                            for role, c in self.configs.items()},
                **({"dialogue": self.dialogue_session.snapshot()} if self.dialogue_session else {})}

    def _finish(self, status, candidates, *, prompt=None, questions=None, reason="", selected=None):
        if self.layer_analysis is not None and status in {"ready", "unreviewed"}:
            selected_model_layers = {d.layer for d in self.layer_decisions if d.choice == "model"}
            unresolved = [layer for layer in self.layer_analysis.missing()
                          if layer.layer in selected_model_layers and layer.needs_confirmation]
            if unresolved:
                status, selected = "needs_clarification", None
                questions = [layer.question for layer in unresolved]
                reason = "模型补全中仍有待确认的事实或权限，当前仅为草稿。"
        if (status in {"ready", "unreviewed"} and prompt
                and adds_reference_ban_after_omission(prompt, self.original, self.layer_decisions)):
            status, selected = "needs_review", None
            reason = "已选择省略参考材料，但候选新增了禁止参考或禁止示例的限制；该草稿未作为最终结果保存。"
        if (status in {"ready", "unreviewed"} and prompt
                and changed_model_reference_block(prompt, self.layer_decisions)):
            status, selected = "needs_review", None
            reason = "候选未完整保留你采用的模型参考块，可能遗漏或改变内容、用途、适用范围或示例映射；该草稿未作为最终结果保存。"
        if (status in {"ready", "unreviewed"} and prompt and self.references is not None
                and any(block not in prompt for block in self.references.blocks)):
            status, selected = "needs_review", None
            reason = "候选未完整保留已加载的文件参考及来源范围；该草稿未作为最终结果保存。"
        if (status in {"ready", "unreviewed"} and self.handoff_context is not None
                and (any(block not in (prompt or "") for block in self.handoff_context.blocks)
                     or self.handoff_context.continuation_block
                     and not (prompt or "").endswith(self.handoff_context.continuation_block))):
            status, selected = "needs_review", None
            reason = "候选未完整保留交接快照，未保存为成功提示词。"
        if self.history_pipeline is not None and self.history_pipeline.report["status"] != "complete":
            self.history_pipeline.report.update(status=status, reason=reason)
            for stage in self.history_pipeline.report["stages"]:
                if stage["status"] == "running":
                    stage["status"] = "interrupted"
                for attempt in stage["attempts"]:
                    if attempt["status"] == "running":
                        attempt.update(status="interrupted", reason="阶段未完成：" + reason)
            for chunk in self.history_pipeline.report["chunks"]:
                if chunk["status"] == "running":
                    chunk["status"] = "incomplete"
        self.run_status = status
        return OptimizationResult(status, prompt, status == "ready", questions or [], reason,
                                  selected, list(self.warnings), [c.to_dict() for c in candidates],
                                  list(self.review_history), self.metadata(),
                                  layers=[layer.model_dump() for layer in self.layer_analysis.layers]
                                  if self.layer_analysis is not None else [],
                                  layer_decisions=[d.model_dump() for d in self.layer_decisions])

    def _decision(self, review, candidates):
        if review.action in {"select", "keep_original"}:
            chosen = next(c for c in candidates if c.id == review.candidate_id)
            return self._finish("ready", candidates, prompt=chosen.draft.optimized_prompt,
                                reason=review.reason, selected=chosen.id)
        if review.action == "needs_clarification":
            return self._finish("needs_clarification", candidates,
                                questions=review.clarification_questions, reason=review.reason)
        return self._finish("needs_review", candidates, reason=review.reason)

    async def _workflow(self):
        outputs = await asyncio.gather(self._generate("a"), self._generate("b"), return_exceptions=True)
        candidates = []
        for role, output in zip(("a", "b"), outputs):
            if isinstance(output, (BudgetExceeded, HandoffError)):
                raise output
            if isinstance(output, BaseException):
                self.warnings.append(f"生成器 {role} 未产出有效候选，本次只比较其余候选和原文。")
            else:
                candidates.append(Candidate("", role, output))
        if not candidates:
            raise OutputError("两个生成器均未产出有效候选。")
        original_prompt = self.references.attach(self.original) if self.references else self.original
        if self.handoff_context is not None:
            original_prompt = self.handoff_context.attach(original_prompt, self.original)
        candidates.append(Candidate("", "original", Draft(
            status="ready", optimized_prompt=original_prompt, preserved_constraints=[],
            clarification_questions=[], change_summary=[])))
        self.rng.shuffle(candidates)
        for index, candidate in enumerate(candidates, 1):
            candidate.id = (f"r{self.dialogue_session.revision}_candidate_{index}"
                            if self.dialogue_session else f"candidate_{index}")
        if self.dialogue_session:
            self._dialogue_candidates = candidates
        review = await self._review(candidates, "review")
        if review.action != "repair":
            return self._decision(review, candidates)
        if not self.options.allow_repair:
            return self._finish("needs_review", candidates, reason="候选需要修复，本次已禁用自动修复。")
        if self.dialogue_session is not None:
            if self.dialogue_session.prompt_repairs_used:
                return self._finish("needs_review", candidates, reason="本会话已使用一次提示词修复，已停止自动修复。")
            self.dialogue_session.prompt_repairs_used += 1
        chosen = next(c for c in candidates if c.id == review.candidate_id)
        assessment = next(r for r in review.reviews if r.candidate_id == chosen.id)
        repaired = await self._generate(chosen.origin, {
            "candidate": chosen.draft.optimized_prompt,
            "findings": [f.model_dump() for f in assessment.findings],
        })
        candidate = Candidate(chosen.id + "_repaired", chosen.origin, repaired)
        candidates.append(candidate)
        finalists = [c for c in candidates if c.origin == "original"] + [candidate]
        self.rng.shuffle(finalists)
        final_review = await self._review(finalists, "repair_review")
        if final_review.action == "repair":
            return self._finish("needs_review", candidates, reason="一次修复后仍未通过，已停止自动修复。")
        return self._decision(final_review, candidates)

    async def run(self, original: str):
        if self._api_mode == "dialogue":
            raise ConfigurationError("同一 Optimizer 不能混用 run 与 run_dialogue。")
        if self.started is not None:
            raise ConfigurationError("每个 Optimizer 实例只能运行一次。")
        if not isinstance(original, str) or not original.strip():
            raise ConfigurationError("原始需求不能为空。")
        if self.options.max_input_chars is not None and len(original) > self.options.max_input_chars:
            raise ConfigurationError("原始需求超过本次允许的字符数。")
        needed = set(self._required_model_roles)
        if not needed.issubset(self.configs):
            raise ConfigurationError("模型角色配置不完整。")
        try:
            workflow, window_data, windows = self._prepare_workflow(self.history, self.context_windows)
        except ConfigurationError as error:
            if isinstance(self.history, HistoryBundle):
                self.workflow, self.original = "handoff", original
                return self._finish("handoff_configuration_error", [], reason=str(error))
            raise
        self._activate_workflow(workflow)
        self.context_windows, self.windows = window_data, windows
        self.original = original
        if (self.options.max_input_chars is not None
                and sum(len(t) for t in self._evidence_texts()) > self.options.max_input_chars):
            raise ConfigurationError("原始需求与文件参考合计超过本次允许的字符数。")
        if self.references is not None:
            self.warnings.extend(self.references.warnings)
        self.started = time.monotonic()
        self._api_mode = "legacy"
        try:
            if self._requires_independent_review:
                self.review_backend = require_review_backend()
            if self.workflow == "handoff":
                if self.history is None:
                    raise ConfigurationError("交接模式需要独立的历史输入。")
                data = self.context_windows
                if isinstance(data, (str, Path)):
                    data = read_window_config(Path(data))
                self.windows = WindowLimits.from_config(data, self.configs)
                self.history_pipeline = HistoryPipeline(self.history, self.windows, self.configs, self.options,
                                                        self._call, progress=self.progress)
                self.handoff_context = await self.history_pipeline.run(original)
                if (self.options.max_input_chars is not None
                        and sum(len(t) for t in self._evidence_texts()) > self.options.max_input_chars):
                    raise WindowExceeded("整理后的任务、上下文及参考超过正式输入字符上限，未裁剪。")
                conflicts = [item["text"] for item in self.handoff_context.payload()["summary"]["items"]
                             if item["status"] == "conflict"]
                if conflicts:
                    return self._finish("needs_clarification", [], questions=conflicts,
                                        reason="交接中存在尚未确认的要求冲突，已保存阶段报告。")
            if self.options.choose_layers:
                pending = await self._prepare_layers()
                if pending is not None:
                    return pending
            return await self._workflow()
        except asyncio.CancelledError:
            if self.workflow == "handoff":
                return self._finish("cancelled", [], reason="交接已取消；阶段报告保留已处理部分，未生成成功提示词。")
            raise
        except BudgetExceeded as error:
            if self.workflow == "handoff":
                return self._finish("budget_exceeded", [], reason=str(error))
            raise
        except WindowExceeded as error:
            return self._finish("context_exceeded", [], reason=str(error))
        except FidelityFailed as error:
            return self._finish("handoff_fidelity_failed", [], reason=str(error))
        except ConfigurationError as error:
            if self.workflow == "handoff":
                return self._finish("handoff_configuration_error", [], reason=str(error))
            raise
        except OptimizerError as error:
            if self.workflow == "handoff":
                return self._finish("handoff_failed", [], reason=str(error))
            raise
        except Exception:
            if self.workflow == "handoff":
                return self._finish("handoff_failed", [], reason="交接发生未预期错误；阶段用量已保留，服务详情未写入报告。")
            raise
        finally:
            try:
                await asyncio.wait_for(asyncio.gather(
                    *(m.close() for m in self.models.values()), return_exceptions=True), timeout=2)
            except Exception:
                # Cleanup failure must not hide a result or a useful primary error.
                pass

    def _emit_dialogue(self, kind, *, phase=None, **data):
        state = self.dialogue_session
        if state is None or self.dialogue_event is None:
            return
        event = {"kind": kind, "phase": phase or self.run_status,
                 "session_id": state.session_id, "revision": state.revision,
                 "request_count": len(self.calls), "known_total_tokens": self.known_tokens,
                 "unknown_usage_requests": self.unknown_usage,
                 "total_tokens": None if self.unknown_usage else self.known_tokens,
                 "elapsed_seconds": round(self._elapsed(), 3),
                 "workflow": self.workflow, "limits": self._limits(),
                 **data}
        try:
            self.dialogue_event(event)
        except Exception:
            pass

    def _pause_dialogue_time(self):
        if self.dialogue_session is not None and self.dialogue_session.paused_at is None:
            self.dialogue_session.paused_at = time.monotonic()

    def _resume_dialogue_time(self):
        state = self.dialogue_session
        if state is not None and state.paused_at is not None:
            elapsed = time.monotonic() - state.paused_at
            self.user_wait_seconds += elapsed
            self.started += elapsed
            state.paused_at = None

    def _refresh_dialogue_request(self):
        state = self.dialogue_session
        self.original = state.confirmed_request
        self.references = self._dialogue_select_references(self.original, self.references)
        state.fingerprint = fingerprint_request(self.original, self.references, self.history, self.context_windows)
        if (self.options.max_input_chars is not None
                and sum(len(text) for text in self._evidence_texts()) > self.options.max_input_chars):
            raise ConfigurationError("已确认需求与参考材料合计超过本次允许的字符数。")

    @staticmethod
    def _dialogue_select_references(text, references):
        if references is not None and getattr(getattr(references, "options", None), "mode", None) == "relevant":
            from optimizer_documents import reselect_references
            return reselect_references(references, query=text)
        return references

    def _publish_dialogue_result(self, result, *, kind="result", batch=None):
        state = self.dialogue_session
        self._pause_dialogue_time()
        request_hash = hashlib.sha256(state.confirmed_request.encode("utf-8")).hexdigest()
        result.candidates = [{**candidate, "session_id": state.session_id, "revision": state.revision,
                              "request_sha256": request_hash, "fingerprint": state.fingerprint}
                             for candidate in result.candidates]
        # Each checkpoint is a detached report snapshot, never a live session view.
        result.metadata = self.metadata()
        archive = {"status": result.status, "optimized_prompt": result.optimized_prompt,
                   "reviewed": result.reviewed, "questions": list(result.questions),
                   "reason": result.reason, "selected_id": result.selected_id,
                   "warnings": list(result.warnings), "candidates": copy.deepcopy(result.candidates),
                   "reviews": copy.deepcopy([review for review in result.reviews
                                             if review.get("revision") == state.revision])}
        # Retain retired handoff evidence without recursively archiving the session.
        archive["metadata"] = copy.deepcopy({key: result.metadata[key] for key in
                                             ("workflow", "limits", "handoff", "context_windows")
                                             if key in result.metadata})
        state.archived_results.append({"revision": state.revision, "fingerprint": state.fingerprint,
                                       "result": archive})
        result.metadata = self.metadata()
        self._emit_dialogue(kind, phase=state.status, result=result, **({"batch": batch} if batch else {}))
        return result

    def _dialogue_pending_result(self, *, reason="", paused=True):
        state = self.dialogue_session
        state.status = "paused" if paused else "waiting"
        result = self._finish("needs_clarification", self._dialogue_candidates,
                              prompt=(self._dialogue_candidates[0].draft.optimized_prompt
                                      if self._dialogue_candidates else None),
                              questions=[question.text for question in state.pending_questions], reason=reason)
        return result

    def _normalize_dialogue_questions(self, questions, *, source="clarification", reason=""):
        state = self.dialogue_session
        by_id = {question.id: question for question in state.pending_questions}
        by_text = {question.text: question for question in state.pending_questions}
        output = []
        for question in questions:
            if isinstance(question, str):
                old = by_text.get(question)
                item = old or DialogueQuestion(id=state.next_question_id(), text=question,
                    reason=reason or "此问题会影响任务要求或验收。", source=source)
            else:
                if isinstance(question, DialogueQuestion):
                    item = question
                else:
                    material = dict(question)
                    material.pop("question_id", None)
                    item = DialogueQuestion.model_validate(material)
                old = None if source == "handoff" else by_id.get(item.id) or by_text.get(item.text)
                if source == "handoff":
                    # A new history snapshot has new item/question identities even
                    # when an unresolved conflict's display text stays the same.
                    item = item.model_copy(update={"source": "handoff"})
                elif old is not None:
                    item = old
                else:
                    if item.related_item_ids:
                        raise OutputError("澄清问题引用了未提供的交接冲突标识。")
                    item = item.model_copy(update={"id": state.next_question_id(), "source": source})
            if not any(existing.id == item.id for existing in output):
                output.append(item)
        return output

    async def _dialogue_clarify(self):
        state = self.dialogue_session
        role = "a"
        payload = {**self._request_payload(), "phase": "dialogue_clarification",
                   "previous_confirmed_request": state.previous_confirmed_request,
                   "latest_updates": list(state.pending_delta),
                   "pending_questions": [question.model_dump() for question in state.pending_questions]}
        system = clarification_prompt()
        for attempt in range(2):
            reply = await self._call(role, system, payload,
                "dialogue_clarification_schema_retry" if attempt else "dialogue_clarification")
            try:
                return parse_output(reply.text, ClarificationDecision)
            except OutputError:
                if attempt:
                    raise
                system += "\n上次输出结构不符合要求，请重新返回完整 JSON；不要猜测用户答案。"

    async def _dialogue_ask(self, on_questions):
        state = self.dialogue_session
        shown = state.pending_questions[:3]
        batch = DialogueQuestionBatch(state.session_id, state.revision, copy.deepcopy(shown),
                                      state.total_rounds + 1)
        checkpoint = self._dialogue_pending_result(reason="等待用户确认关键问题。", paused=False)
        self._publish_dialogue_result(checkpoint, kind="questions", batch=batch)
        if on_questions is None:
            state.status = "paused"
            return self._dialogue_pending_result(reason="尚未提供回答，可保存会话记录后继续。")
        try:
            value = on_questions(batch)
            if inspect.isawaitable(value):
                value = await value
            response = normalize_response(value)
        finally:
            self._resume_dialogue_time()
        if response.action == "cancel":
            raise asyncio.CancelledError()
        if response.action == "pause":
            return self._dialogue_pending_result(reason="用户已暂停；已确认需求、未决项及已有草稿已保留。")
        by_id = {question.id: question for question in shown}
        accepted = []
        for answer in response.answers:
            question = by_id.get(answer.question_id)
            if question is None:
                raise ConfigurationError("回答对应的问题不在当前问题列表中。")
            option = None
            if answer.option_id is not None:
                option = next((item for item in question.options if item.id == answer.option_id), None)
                if option is None:
                    raise ConfigurationError("选择的建议不在已展示选项中。")
            text = answer.text
            if option is not None:
                text = option.label + ("\n用户补充：" + answer.text if answer.text.strip() else "")
            if not text.strip():
                continue
            accepted.append((answer, question, text))
        state.question_rounds.append({**batch.to_dict(), "response": response.model_dump()})
        if not accepted:
            return self._dialogue_pending_result(reason="尚无有效回答，未默认采用任何模型建议。")
        preview = copy.deepcopy(state)
        for answer, question, text in accepted:
            preview.append(text, question=question, option_id=answer.option_id, raw_text=answer.text)
        try:
            selected = self._dialogue_select_references(preview.confirmed_request, self.references)
            if (self.options.max_input_chars is not None
                    and len(preview.confirmed_request) + sum(len(text) for text in (
                        selected.blocks if selected is not None else [])) > self.options.max_input_chars):
                raise ConfigurationError("回答与已有确认需求、参考材料合计超过允许字符数，请缩短回答后继续。")
        except ConfigurationError as error:
            state.question_rounds[-1].update(accepted=False, error=str(error))
            return self._dialogue_pending_result(reason=str(error))
        self.references = selected
        state.question_rounds[-1]["accepted"] = True
        state.total_rounds += 1
        previous = state.confirmed_request
        for answer, question, text in accepted:
            state.append(text, question=question, option_id=answer.option_id, raw_text=answer.text)
        state.previous_confirmed_request = previous
        answered_ids = {answer.question_id for answer, _, _ in accepted}
        state.pending_questions = [question for question in state.pending_questions if question.id not in answered_ids]
        self._dialogue_candidates = []
        self._refresh_dialogue_request()
        return None

    async def _prepare_dialogue_handoff(self, change_kind):
        from optimizer_handoff import DialogueHandoffPipeline, history_fingerprint, handoff_conflict_questions
        state = self.dialogue_session
        if self.history is None:
            raise ConfigurationError("交接模式需要独立的历史输入。")
        data = self.context_windows
        if isinstance(data, (str, Path)):
            data = read_window_config(Path(data))
        self.windows = WindowLimits.from_config(data, self.configs)
        identity = history_fingerprint(self.history)
        if self._dialogue_handoff_identity == identity and self._dialogue_last_handoff_pipeline is not None:
            pipeline = self._dialogue_last_handoff_pipeline
            pipeline.windows = self.windows
            pipeline.options = self.options
        else:
            pipeline = DialogueHandoffPipeline(self.history, self.windows, self.configs, self.options, self._call,
                progress=lambda message: self._emit_dialogue("stage", phase="handoff", message=message))
        self.history_pipeline = pipeline
        unchanged = (self._dialogue_handoff_state is not None and identity == self._dialogue_handoff_identity
                     and self._dialogue_handoff_state.snapshot.prompt_payload()["current_request_sha256"]
                     == hashlib.sha256(self.original.encode("utf-8")).hexdigest())
        if unchanged:
            self.history_pipeline = self._dialogue_last_handoff_pipeline or pipeline
            return self._normalize_dialogue_questions(handoff_conflict_questions(self._dialogue_handoff_state),
                                                     source="handoff")
        if (self._dialogue_handoff_state is not None and identity == self._dialogue_handoff_identity
                and change_kind in {"detail", "conflict_resolution"}):
            handoff = await pipeline.update(self._dialogue_handoff_state, self.original,
                                           list(state.pending_delta), revision=state.revision)
        else:
            handoff = await pipeline.full(self.original, revision=state.revision)
        self._dialogue_handoff_state = handoff
        self._dialogue_handoff_identity = identity
        self._dialogue_last_handoff_pipeline = pipeline
        self.handoff_context = handoff.snapshot
        self._refresh_dialogue_request()
        state.pending_delta.clear()
        questions = handoff_conflict_questions(handoff)
        return self._normalize_dialogue_questions(questions, source="handoff")

    async def run_dialogue(self, request, on_questions=None):
        """Run/continue one session, retaining budgets and clients between submissions."""
        if self._api_mode == "legacy":
            raise ConfigurationError("同一 Optimizer 不能混用 run 与 run_dialogue。")
        if self._closed or self.dialogue_session is not None and self.dialogue_session.closed:
            raise ConfigurationError("此会话已结束，请创建新会话。")
        if self._dialogue_busy:
            raise ConfigurationError("当前会话正在运行，请等待完成后再提交。")
        if self._dialogue_options is not None and (self.options != self._dialogue_options
                                                    or self.configs != self._dialogue_configs):
            raise ConfigurationError("同一会话的预算及模型配置不能更换，请创建新会话。")
        request = DialogueRequest(request) if isinstance(request, str) else request
        if not isinstance(request, DialogueRequest) or not isinstance(request.text, str) or not request.text.strip():
            raise ConfigurationError("原始需求不能为空。")
        if self.options.max_input_chars is not None and len(request.text) > self.options.max_input_chars:
            raise ConfigurationError("原始需求超过本次允许的字符数。")
        resuming = request.resume or request.continue_rounds
        if resuming and (self.dialogue_session is None or self.dialogue_session.status != "paused"):
            raise ConfigurationError("只有暂停的会话可以恢复对话。")
        # Validate a detached preview before changing a usable session or its revision.
        preview = copy.deepcopy(self.dialogue_session) if self.dialogue_session is not None else DialogueSession()
        if not resuming and (not preview.original_request or request.text != preview.confirmed_request):
            preview.append(request.text)
        preview_references = self.references if request.references is UNSET else request.references
        preview_references = self._dialogue_select_references(preview.confirmed_request, preview_references)
        preview_history = copy.deepcopy(self.history if request.history is UNSET else request.history)
        preview_windows = self.context_windows if request.context_windows is UNSET else request.context_windows
        if preview_history is None and request.context_windows is UNSET:
            preview_windows = None
        workflow, window_data, windows = self._prepare_workflow(preview_history, preview_windows)
        if not set(self._required_model_roles).issubset(self.configs):
            raise ConfigurationError("模型角色配置不完整。")
        old_history_identity = history_fingerprint(self.history) if self.history is not None else None
        new_history_identity = history_fingerprint(preview_history) if preview_history is not None else None
        preview_texts = [preview.confirmed_request] + (list(preview_references.blocks)
                                                       if preview_references is not None else [])
        if (self.options.max_input_chars is not None
                and sum(len(text) for text in preview_texts) > self.options.max_input_chars):
            raise ConfigurationError("已确认需求与新补充、参考材料合计超过本次允许的字符数。")
        self._dialogue_busy = True
        self._api_mode = "dialogue"
        if self.dialogue_session is None:
            self.dialogue_session = DialogueSession()
            self._dialogue_options = self.options
            self._dialogue_configs = copy.deepcopy(self.configs)
            self.started = time.monotonic()
        else:
            self._resume_dialogue_time()
        state = self.dialogue_session
        state.status = "running"
        self.run_status = "running"
        try:
            previous_fingerprint = state.fingerprint
            old_material = fingerprint_request("", self.references, self.history, self.context_windows)
            self.references = copy.deepcopy(preview_references)
            self.history = preview_history
            self.context_windows = window_data
            if old_history_identity != new_history_identity or self.workflow == "handoff" and workflow == "normal":
                self._clear_active_handoff()
            self._activate_workflow(workflow)
            self.windows = windows
            new_material = fingerprint_request("", self.references, self.history, self.context_windows)
            previous_revision = state.revision
            if not resuming and (not state.original_request or request.text != state.confirmed_request):
                state.append(request.text)
            if old_material != new_material and state.original_request and state.revision == previous_revision:
                state.revision += 1
            self.handoff_context = (self._dialogue_handoff_state.snapshot if self._dialogue_handoff_state else None)
            self._refresh_dialogue_request()
            if previous_fingerprint != state.fingerprint:
                self._dialogue_candidates = []
            if self._requires_independent_review and self.review_backend is None:
                self.review_backend = require_review_backend()
            if self.references is not None:
                for warning in self.references.warnings:
                    if warning not in self.warnings:
                        self.warnings.append(warning)
            if self.workflow == "handoff" and (self._dialogue_handoff_state is None
                    or self.history is None or self._dialogue_handoff_identity != history_fingerprint(self.history)):
                conflicts = await self._prepare_dialogue_handoff("uncertain")
                state.pending_delta.clear()
                state.pending_questions = conflicts
                if conflicts:
                    pending = await self._dialogue_ask(on_questions)
                    if pending is not None:
                        return self._publish_dialogue_result(pending)
            elif resuming and previous_fingerprint == state.fingerprint and state.pending_questions:
                pending = await self._dialogue_ask(on_questions)
                if pending is not None:
                    return self._publish_dialogue_result(pending)
            while True:
                decision = await self._dialogue_clarify()
                state.change_kind = decision.change_kind
                if self.workflow == "handoff":
                    conflicts = await self._prepare_dialogue_handoff(decision.change_kind)
                    if conflicts:
                        state.pending_questions = conflicts
                        pending = await self._dialogue_ask(on_questions)
                        if pending is not None:
                            return self._publish_dialogue_result(pending)
                        continue
                if decision.status == "ask":
                    state.pending_questions = self._normalize_dialogue_questions(decision.questions)
                    pending = await self._dialogue_ask(on_questions)
                    if pending is not None:
                        return self._publish_dialogue_result(pending)
                    continue
                state.pending_questions = []
                state.pending_delta.clear()
                result = await self._workflow()
                if result.status == "needs_clarification":
                    state.pending_questions = self._normalize_dialogue_questions(result.questions,
                        source="judge", reason=result.reason)
                    pending = await self._dialogue_ask(on_questions)
                    if pending is not None:
                        return self._publish_dialogue_result(pending)
                    continue
                state.status = "open"
                return self._publish_dialogue_result(result)
        except asyncio.CancelledError:
            state.status = "cancelled"
            result = self._finish("cancelled", self._dialogue_candidates,
                                  reason="会话已取消，已确认需求、未决项和已有草稿保留于报告。")
        except BudgetExceeded as error:
            state.status = "budget_exceeded"
            result = self._finish("budget_exceeded", self._dialogue_candidates,
                                  reason=str(error) if str(error) else "已达到本会话 token 调度预算。")
        except (OptimizerError, Exception) as error:
            state.status = "failed"
            status = ("context_exceeded" if isinstance(error, WindowExceeded) else
                      "handoff_fidelity_failed" if isinstance(error, FidelityFailed) else "failed")
            reason = str(error) if isinstance(error, OptimizerError) else "会话运行未完成，服务详情未写入报告。"
            result = self._finish(status, self._dialogue_candidates, reason=reason)
        finally:
            self._dialogue_busy = False
            self._pause_dialogue_time()
            if state.status in {"cancelled", "failed", "budget_exceeded"}:
                await self.aclose()
        return self._publish_dialogue_result(result)

    async def cancel_dialogue(self):
        if self._dialogue_busy:
            raise ConfigurationError("请通过 Controller 取消正在运行的会话。")
        state = self.dialogue_session
        if state is None or state.closed:
            return None
        state.status = "cancelled"
        result = self._finish("cancelled", self._dialogue_candidates, reason="会话已取消，阶段报告已保留。")
        await self.aclose()
        return self._publish_dialogue_result(result)

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        if self.dialogue_session is not None:
            self._pause_dialogue_time()
            self.dialogue_session.closed = True
            if self.dialogue_session.status == "open":
                self.dialogue_session.status = "closed"
        try:
            await asyncio.wait_for(asyncio.gather(
                *(model.close() for model in self.models.values()), return_exceptions=True), timeout=2)
        except Exception:
            pass
