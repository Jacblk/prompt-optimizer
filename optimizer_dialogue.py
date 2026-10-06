"""UI-independent dialogue contracts and the owner of a bounded session."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
import hashlib
import inspect
import json
from typing import Literal, TYPE_CHECKING
from uuid import uuid4

from pydantic import Field, model_validator

from optimizer_config import ConfigurationError
from optimizer_models import StrictModel, Text

if TYPE_CHECKING:
    from optimizer_engine import OptimizationResult, Optimizer


UNSET = object()


class DialogueOption(StrictModel):
    id: Text
    label: Text


class DialogueQuestion(StrictModel):
    id: Text
    text: Text
    reason: Text
    options: list[DialogueOption] = Field(default_factory=list, max_length=5)
    source: Literal["clarification", "judge", "handoff", "generator"] = "clarification"
    related_item_ids: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_options(self):
        if len({option.id for option in self.options}) != len(self.options):
            raise ValueError("Option IDs must be unique")
        return self


class DialogueAnswer(StrictModel):
    question_id: Text
    text: str = ""
    option_id: str | None = None


class DialogueResponse(StrictModel):
    action: Literal["answer", "pause", "cancel"] = "answer"
    answers: list[DialogueAnswer] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def consistent_answers(self):
        if self.action != "answer" and self.answers:
            raise ValueError("Only an answer response can contain answers")
        if len({answer.question_id for answer in self.answers}) != len(self.answers):
            raise ValueError("A question can only be answered once in a batch")
        return self


@dataclass(frozen=True)
class DialogueQuestionBatch:
    session_id: str
    revision: int
    questions: list[DialogueQuestion]
    round_number: int
    paused: bool = False
    reason: str = ""

    def to_dict(self):
        return copy.deepcopy({"session_id": self.session_id, "revision": self.revision,
                "questions": [question.model_dump() for question in self.questions],
                "round_number": self.round_number,
                "paused": self.paused, "reason": self.reason})


class ClarificationDecision(StrictModel):
    status: Literal["ask", "sufficient"]
    questions: list[DialogueQuestion] = Field(default_factory=list, max_length=3)
    change_kind: Literal["detail", "conflict_resolution", "goal_change", "scope_expansion", "uncertain"] = "uncertain"
    reason: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def consistent_status(self):
        if bool(self.questions) != (self.status == "ask"):
            raise ValueError("Clarification status and questions disagree")
        if len({question.id for question in self.questions}) != len(self.questions):
            raise ValueError("Question IDs must be unique")
        return self


@dataclass(frozen=True)
class DialogueRequest:
    text: str
    # Legacy positional/keyword argument: now an alias for resume.
    continue_rounds: bool = False
    references: object = UNSET
    history: object = UNSET
    context_windows: object = UNSET
    resume: bool = False


@dataclass
class DialogueSession:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    revision: int = 0
    original_request: str = ""
    confirmed_request: str = ""
    previous_confirmed_request: str = ""
    updates: list[dict] = field(default_factory=list)
    question_rounds: list[dict] = field(default_factory=list)
    pending_questions: list[DialogueQuestion] = field(default_factory=list)
    pending_delta: list[dict] = field(default_factory=list)
    archived_results: list[dict] = field(default_factory=list)
    total_rounds: int = 0
    prompt_repairs_used: int = 0
    question_counter: int = 0
    fingerprint: str = ""
    status: str = "open"
    paused_at: float | None = None
    closed: bool = False
    change_kind: str = "uncertain"

    def next_question_id(self):
        self.question_counter += 1
        return f"q{self.question_counter}"

    def append(self, text, *, question=None, option_id=None, raw_text=None):
        self.previous_confirmed_request = self.confirmed_request
        if not self.original_request:
            self.original_request = text
        else:
            entry = {"revision": self.revision + 1, "text": text,
                     "raw_text": text if raw_text is None else raw_text,
                     "question_id": question.id if question else f"update-r{self.revision + 1}",
                     "question": question.text if question else None,
                     "option_id": option_id,
                     "related_item_ids": list(question.related_item_ids) if question else []}
            self.updates.append(entry)
            self.pending_delta.append({**{key: entry[key] for key in
                                       ("question_id", "text", "option_id", "related_item_ids")},
                                       "question": question.text if question else "用户主动补充或纠正需求"})
        self.revision += 1
        self.confirmed_request = self.original_request
        if self.updates:
            self.confirmed_request += ("\n\n## 用户确认的补充与纠正\n"
                "以下按确认时间排列；后续明确纠正替代对应旧要求，其余原有要求和约束继续有效。\n"
                "对应问题仅记录提问对象，不是已确认事实或授权；只有用户回答和明确采用的选项是新增需求。\n")
            for index, update in enumerate(self.updates, 1):
                if update["question"]:
                    self.confirmed_request += f"\n补充 {index} 对应问题：{update['question']}\n"
                else:
                    self.confirmed_request += f"\n补充 {index}：\n"
                if update["option_id"]:
                    self.confirmed_request += "用户明确采用建议："
                self.confirmed_request += update["text"] + "\n"

    def snapshot(self):
        return copy.deepcopy({"session_id": self.session_id, "revision": self.revision,
                "fingerprint": self.fingerprint, "status": self.status,
                "closed": self.closed, "original_request": self.original_request,
                "confirmed_request": self.confirmed_request,
                "updates": [dict(update) for update in self.updates],
                "question_rounds": [dict(round_) for round_ in self.question_rounds],
                "pending_questions": [question.model_dump() for question in self.pending_questions],
                "total_rounds": self.total_rounds,
                "prompt_repairs_used": self.prompt_repairs_used,
                "archived_results": list(self.archived_results)})


def fingerprint_request(text, references=None, history=None, context_windows=None):
    reference_data = ({"metadata": references.metadata(), "blocks": references.blocks}
                      if references is not None else None)
    if history is not None:
        from optimizer_handoff import history_fingerprint
        history_data = {"sources": [source.summary() for source in history.sources],
                        "fingerprint": history_fingerprint(history)}
    else:
        history_data = None
    windows = (context_windows.metadata() if hasattr(context_windows, "metadata") else context_windows)
    material = {"confirmed_request": text, "references": reference_data,
                "history": history_data, "context_windows": windows}
    return hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True,
                                    default=str).encode("utf-8")).hexdigest()


def normalize_response(value):
    try:
        if isinstance(value, DialogueResponse):
            return value
        if value is None:
            return DialogueResponse(action="pause")
        if isinstance(value, list):
            return DialogueResponse(answers=value)
        return DialogueResponse.model_validate(value)
    except (TypeError, ValueError):
        raise ConfigurationError("澄清回答结构不完整，请按对应问题回答或明确暂停。") from None


class DialogueController:
    """Own one optimizer; the UI edits drafts and submits immutable snapshots."""

    def __init__(self, optimizer: Optimizer, *, on_questions=None, on_event=None):
        self.optimizer = optimizer
        self.on_questions = on_questions
        self.on_event = on_event
        self.result: OptimizationResult | None = None
        self._task = None
        self.optimizer.dialogue_event = self._event

    def _event(self, event):
        if event.get("kind") in {"questions", "result"} and event.get("result") is not None:
            self.result = event["result"]
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                # A view or report notification must not change model accounting.
                pass

    async def _questions(self, batch):
        if self.on_questions is None:
            return DialogueResponse(action="pause")
        value = self.on_questions(batch)
        if inspect.isawaitable(value):
            value = await value
        return normalize_response(value)

    async def _submit(self, request):
        if self._task is not None and not self._task.done():
            raise ConfigurationError("当前会话正在运行，请等待完成后再提交。")
        self._task = asyncio.current_task()
        try:
            self.result = await self.optimizer.run_dialogue(request, self._questions)
            return self.result
        finally:
            self._task = None

    async def submit(self, text: str, *, references=UNSET, history=UNSET, context_windows=UNSET):
        return await self._submit(DialogueRequest(text, references=references, history=history,
                                                 context_windows=context_windows))

    async def resume(self):
        state = self.optimizer.dialogue_session
        if state is None or state.status != "paused":
            raise ConfigurationError("只有暂停的会话可以恢复对话。")
        return await self._submit(DialogueRequest(state.confirmed_request, resume=True))

    async def continue_rounds(self):
        """Compatibility alias; resuming no longer changes a round allowance."""
        return await self.resume()

    async def cancel(self):
        task = self._task
        if task is asyncio.current_task() and not task.done():
            raise asyncio.CancelledError()
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        else:
            self.result = await self.optimizer.cancel_dialogue()
        return self.result

    async def close(self):
        if self._task is not None and not self._task.done():
            await self.cancel()
        await self.optimizer.aclose()

    def snapshot(self):
        state = self.optimizer.dialogue_session
        return state.snapshot() if state is not None else {}

    def is_current(self, result):
        state = self.optimizer.dialogue_session
        if result is None or state is None:
            return False
        info = result.metadata.get("dialogue", {})
        return (not state.closed and info.get("session_id") == state.session_id and info.get("revision") == state.revision
                and info.get("fingerprint") == state.fingerprint)
