"""Conversational terminal UI. Import/start/preview/report reading are offline."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Footer, Header, Input, Label, OptionList, RichLog, Select,
    Static, TabbedContent, TabPane, TextArea,
)

from optimizer_config import ROOT, ConfigurationError, OptimizerError, load_models, read_environment
from optimizer_documents import DEFAULT_PURPOSE, DEFAULT_USAGE, ReferenceFile, ReferenceOptions, check_reference_path, prepare_references
from optimizer_dialogue import UNSET
from optimizer_engine import OptimizationResult, Optimizer, RunOptions
from optimizer_handoff import WindowLimits, prepare_history, read_window_config
from optimizer_settings import default_settings, read_settings, run_options, reference_options, history_options
from tui_clipboard import copy_windows_text


STATUS_LABELS = {
    "ready": "评审通过", "unreviewed": "生成完成 · 未经独立评审",
    "needs_clarification": "待确认", "needs_review": "需要复核", "cancelled": "已取消",
    "failed": "运行失败", "budget_exceeded": "预算不足", "context_exceeded": "关键内容超限",
    "handoff_failed": "交接未完成", "handoff_fidelity_failed": "交接保真核验失败",
    "handoff_output_truncated": "交接输出被截断",
    "handoff_configuration_error": "交接配置错误",
}
PHASE_LABELS = {
    "idle": "等待输入", "preparing": "准备需求与材料", "clarification": "判断关键缺口",
    "questions": "等待回答", "paused": "已暂停", "generation": "生成提示词",
    "generating": "生成提示词", "review": "独立评审", "repair": "修复提示词",
    "handoff": "整理旧记录", "handoff_extract": "整理旧记录", "handoff_merge": "归并旧记录",
    "preview": "预览材料", "cancelled": "已取消", "finished": "本轮结束",
}
ACTIVITY_LABELS = {
    "waiting": "等待响应", "thinking": "正在思考", "receiving": "正在接收结果",
    "validating": "正在校验", "retrying": "正在重试",
}
SHORT_ACTIVITY_LABELS = {
    "waiting": "等待", "thinking": "思考", "receiving": "接收",
    "validating": "校验", "retrying": "重试",
}


def _value(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _load_configs(root):
    # Configuration is loaded only after an explicit send, never for viewing.
    return load_models(read_environment(root / ".env"))


def _save_result(result, **kwargs):
    from optimizer_reports import save_dialogue_result
    return save_dialogue_result(result, **kwargs)


def _publish_prompt(result, **kwargs):
    from optimizer_reports import publish_dialogue_prompt
    return publish_dialogue_prompt(result, **kwargs)


def _load_report(path):
    from optimizer_reports import load_report
    return load_report(path)


def _save_draft(text, path, **kwargs):
    from optimizer_reports import save_draft
    return save_draft(text, path, **kwargs)


def _make_controller(optimizer, **kwargs):
    from optimizer_dialogue import DialogueController
    return DialogueController(optimizer, **kwargs)


def render_report(data):
    """Read both current and old reports without reconstructing a model session."""
    lines = [STATUS_LABELS.get(data.get("status"), str(data.get("status", "未知状态")))]
    if data.get("reason"):
        lines += ["", str(data["reason"])]
    for question in data.get("questions", []):
        lines.append("待确认：" + str(question))
    if data.get("optimized_prompt"):
        lines += ["", "提示词", str(data["optimized_prompt"])]
    for candidate in data.get("candidates", []):
        if isinstance(candidate, dict):
            lines += ["", "候选 " + str(candidate.get("id", "")), str(candidate.get("optimized_prompt", ""))]
            for change in candidate.get("change_summary", []):
                lines.append("改写说明（生成器自述）：" + str(change))
    for decision in data.get("layer_decisions", []):
        if isinstance(decision, dict):
            lines += ["", "补充选择：" + str(decision.get("layer", "")), str(decision.get("value", ""))]
    for index, review in enumerate(data.get("reviews", []), 1):
        if not isinstance(review, dict):
            continue
        lines += ["", f"第 {index} 次评审：{review.get('reason', '')}"]
        for assessment in review.get("reviews", []):
            lines.append(f"{assessment.get('candidate_id', '')}：{assessment.get('verdict', '')}；{assessment.get('reason', '')}")
            for finding in assessment.get("findings", []):
                lines += ["问题：" + str(finding.get("explanation", "")), "原文依据：" + str(finding.get("source_quote", ""))]
    metadata = data.get("metadata") or {}
    tokens = metadata.get("total_tokens")
    lines += ["", f"请求尝试：{metadata.get('request_count', '未知')}；总 token：{tokens if tokens is not None else '未知'}"]
    references = metadata.get("reference_files") or {}
    for file in references.get("files", []):
        lines.append(f"材料：{file.get('source', '')}；{file.get('selected_chunks', 0)}/{file.get('total_chunks', 0)} 块；{file.get('mode', '')}")
    for block in references.get("blocks", []):
        lines += ["", str(block)]
    handoff = metadata.get("handoff") or {}
    if handoff:
        lines += ["", "旧记录整理状态：" + str(handoff.get("status", "未知"))]
        snapshot = handoff.get("snapshot") or {}
        if snapshot.get("context_block"):
            lines.append(str(snapshot["context_block"]))
    for warning in data.get("warnings", []):
        lines.append("提示：" + str(warning))
    return "\n".join(lines)


def render_dialogue(data):
    dialogue = (data.get("metadata") or {}).get("dialogue")
    if not dialogue:
        return "这份旧报告没有会话记录。结果、候选与评审仍可查看。"
    lines = [f"会话版本：{dialogue.get('revision', '未知')}；状态：{dialogue.get('status', '未知')}",
             "", "原始需求", str(dialogue.get("original_request", "")),
             "", "完整已确认需求", str(dialogue.get("confirmed_request", ""))]
    for index, update in enumerate(dialogue.get("updates", []), 1):
        lines += ["", f"补充 / 纠正 {index}"]
        if update.get("question"):
            lines.append("对应问题：" + str(update["question"]))
        if update.get("option_id"):
            lines.append("明确采用的建议：" + str(update.get("text", "")))
        lines.append("用户回答原文：\n" + str(update.get("raw_text", update.get("text", ""))))
    for question in dialogue.get("pending_questions", []):
        lines += ["", "尚未确认：" + str(question.get("text", ""))]
        if question.get("reason"):
            lines.append("提问理由：" + str(question["reason"]))
        for option in question.get("options", []):
            lines.append("可选建议（未采用）：" + str(option.get("label", "")))
    return "\n".join(lines)


class QuestionScreen(ModalScreen[dict]):
    BINDINGS = [Binding("escape", "pause", "暂停并保存"), Binding("f2", "answer", "提交回答", priority=True)]

    def __init__(self, batch):
        super().__init__()
        self.batch = batch
        self.questions = list(_value(batch, "questions", []))
        if len(self.questions) > 3:
            raise ValueError("每轮最多三个问题")

    def compose(self) -> ComposeResult:
        with Vertical(id="question-dialog", classes="dialog"):
            yield Label(f"需要确认 · 第 {_value(self.batch, 'round_number', 1)} 轮", classes="dialog-title")
            yield Static("建议选项不会自动采用；可以只回答其中一部分。", classes="question-reason")
            with VerticalScroll(id="questions-scroll"):
                for index, question in enumerate(self.questions):
                    yield Label(f"{index + 1}. {_value(question, 'text', '')}", classes="question-label")
                    if _value(question, "reason", ""):
                        yield Static(_value(question, "reason"), classes="question-reason")
                    options = [(_value(option, "label"), _value(option, "id")) for option in _value(question, "options", [])]
                    if options:
                        yield Select(options, prompt="不采用建议（可在下方自由回答）", allow_blank=True, id=f"answer-option-{index}")
                    yield TextArea(id=f"answer-text-{index}", classes="question-answer")
            yield Static("", id="question-error", classes="modal-error")
            with Horizontal(classes="button-row"):
                yield Button("提交回答 F2", id="answer-submit", variant="primary")
                yield Button("暂停并保存", id="answer-pause")
                yield Button("取消会话", id="answer-cancel", variant="error")

    def action_answer(self):
        answers = []
        for index, question in enumerate(self.questions):
            text = self.query_one(f"#answer-text-{index}", TextArea).text
            options = self.query(f"#answer-option-{index}")
            selected = options.first(Select).value if options else Select.NULL
            option_id = None if selected is Select.NULL else selected
            if text.strip() or option_id is not None:
                answers.append({"question_id": _value(question, "id"), "text": text, "option_id": option_id})
        if not answers:
            self.query_one("#question-error", Static).update("请至少回答一项，或选择暂停。")
            return
        self.dismiss({"action": "answer", "answers": answers})

    def action_pause(self):
        self.dismiss({"action": "pause", "answers": []})

    @on(Button.Pressed)
    def pressed(self, event):
        event.stop()
        if event.button.id == "answer-submit":
            self.action_answer()
        elif event.button.id == "answer-pause":
            self.action_pause()
        elif event.button.id == "answer-cancel":
            self.dismiss({"action": "cancel", "answers": []})


class PathScreen(ModalScreen[Path | None]):
    BINDINGS = [Binding("escape", "cancel", "取消")]

    def __init__(self, title, default):
        super().__init__()
        self.title_text, self.default = title, default

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            yield Input(str(self.default), id="save-path")
            yield Static("", id="path-error", classes="modal-error")
            with Horizontal(classes="button-row"):
                yield Button("保存", id="path-save", variant="primary")
                yield Button("取消", id="path-cancel")

    def action_cancel(self):
        self.dismiss(None)

    @on(Button.Pressed)
    def pressed(self, event):
        event.stop()
        if event.button.id == "path-cancel":
            self.action_cancel()
        elif event.button.id == "path-save":
            value = self.query_one("#save-path", Input).value.strip().strip('"')
            if value:
                self.dismiss(Path(value).expanduser())
            else:
                self.query_one("#path-error", Static).update("请填写保存路径。")


class PreviewScreen(ModalScreen):
    BINDINGS = [Binding("escape", "close", "关闭")]

    def __init__(self, text):
        super().__init__()
        self.preview_text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="preview-dialog", classes="dialog"):
            yield Label("材料预览 · 来源与范围", classes="dialog-title")
            yield TextArea(self.preview_text, id="preview-text", read_only=True)
            yield Button("关闭", id="preview-close")

    def action_close(self):
        self.dismiss()

    @on(Button.Pressed, "#preview-close")
    def pressed(self, event):
        event.stop()
        self.action_close()


class ReportScreen(ModalScreen):
    BINDINGS = [Binding("escape", "close", "关闭")]

    def __init__(self, path, report):
        super().__init__()
        self.path, self.report = path, report

    def compose(self) -> ComposeResult:
        with Vertical(id="report-dialog", classes="dialog"):
            yield Label("报告回看 · " + self.path.name, classes="dialog-title")
            with TabbedContent():
                with TabPane("结果与评审"):
                    yield TextArea(render_report(self.report), id="report-text", read_only=True)
                with TabPane("会话记录"):
                    yield TextArea(render_dialogue(self.report), id="report-dialogue", read_only=True)
            yield Button("关闭", id="report-close")

    def action_close(self):
        self.dismiss()

    @on(Button.Pressed, "#report-close")
    def pressed(self, event):
        event.stop()
        self.action_close()


class ReportPickerScreen(ModalScreen[Path | None]):
    BINDINGS = [Binding("escape", "cancel", "取消")]

    def __init__(self, root):
        super().__init__()
        try:
            self.paths = sorted((root / "reports").glob("*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)[:30]
        except OSError:
            self.paths = []

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("离线查看报告", classes="dialog-title")
            yield OptionList(*[path.name for path in self.paths], id="recent-reports")
            yield Input(placeholder="或粘贴完整 JSON 报告路径", id="report-path")
            with Horizontal(classes="button-row"):
                yield Button("打开", id="report-open", variant="primary")
                yield Button("取消", id="report-picker-cancel")

    def action_cancel(self):
        self.dismiss(None)

    @on(OptionList.OptionSelected, "#recent-reports")
    def selected(self, event):
        self.query_one("#report-path", Input).value = str(self.paths[event.option_index])

    @on(Button.Pressed)
    def pressed(self, event):
        event.stop()
        if event.button.id == "report-picker-cancel":
            self.action_cancel()
        elif event.button.id == "report-open":
            value = self.query_one("#report-path", Input).value.strip().strip('"')
            options = self.query_one("#recent-reports", OptionList)
            if value:
                self.dismiss(Path(value).expanduser())
            elif options.highlighted is not None and self.paths:
                self.dismiss(self.paths[options.highlighted])


class _PreparationCancelled(Exception):
    pass


class _MissingWindows(ConfigurationError):
    pass


@dataclass(frozen=True)
class _Submission:
    text: str
    materials: tuple[ReferenceFile, ...]
    reference_mode: str
    history_source: str
    history_text: str
    material_version: int
    token_budget_text: str

    @property
    def workflow(self):
        return "handoff" if self.history_text.strip() else "normal"

    @property
    def reference_signature(self):
        return (self.reference_mode,
                tuple((material.path, material.purpose, material.usage) for material in self.materials))

    @property
    def history_signature(self):
        return (self.workflow, self.history_source, self.history_text) if self.workflow == "handoff" else ("normal",)


class OptimizerApp(App):
    TITLE = "对话式提示词优化器"
    SUB_TITLE = "仅生成提示词"
    CSS_PATH = "tui.tcss"
    BINDINGS = [
        Binding("f2", "send", "发送", priority=True),
        Binding("f3", "toggle_sidebar", "设置/材料/历史"),
        Binding("ctrl+n", "new_session", "新会话"),
        Binding("ctrl+r", "reports", "报告"),
        Binding("ctrl+q", "quit", "退出"),
    ]

    def __init__(self, *, root=ROOT, controller_factory=None, optimizer_factory=Optimizer,
                 config_loader=None, report_saver=None, report_loader=None,
                 draft_saver=None, clipboard_writer=None, prompt_saver=None, configuration_store_factory=None):
        super().__init__()
        self.root = Path(root).resolve()
        self.controller_factory = controller_factory or _make_controller
        self.optimizer_factory = optimizer_factory
        self.config_loader = config_loader or _load_configs
        self.report_saver = report_saver or _save_result
        self.prompt_saver = prompt_saver or _publish_prompt
        self.report_loader = report_loader or _load_report
        self.draft_saver = draft_saver or _save_draft
        self.configuration_store_factory = configuration_store_factory
        self._pending_configuration = False
        self._configuration_stamp = None
        self._runtime_settings = default_settings()
        self.clipboard_writer = clipboard_writer
        self.controller = None
        self.result = None
        self.busy = False
        self.paused = False
        self.materials = []
        self.material_version = 0
        self._submitted_material_version = 0
        self._submitted_reference_signature = None
        self._submitted_history_signature = None
        self._report_keys = set()
        self._report_lock = asyncio.Lock()
        self._report_workers = []
        self._worker = None
        self._cancel_requested = False
        self._quit_requested = False
        self._preview_active = False
        self._core_active = False
        self._provisional_session_id = str(uuid4())
        self._submitted_text = ""
        self._last_phase = "idle"
        self._status_timer = None
        self._activity_request_ids = {}
        self._finished_activity_requests = set()
        self.report_paths = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="workspace"):
            with Vertical(id="conversation"):
                yield Label("对话记录", classes="section-label")
                yield RichLog(id="chat-log", wrap=True, markup=False, highlight=False)
                yield Label("提示词结果 / 草稿（只读）", classes="section-label")
                yield TextArea(
                    id="prompt-output", read_only=True, show_cursor=False,
                    highlight_cursor_line=False,
                    placeholder="生成结果显示在这里；请在下方输入需求。",
                )
                with Horizontal(classes="button-row"):
                    yield Button("复制", id="copy-prompt")
                    yield Button("保存提示词", id="save-prompt")
                    yield Button("另存草稿", id="save-draft")
                    yield Button("回看报告", id="view-report")
                yield Label("输入需求或补充 · Enter 换行，F2 或发送提交", classes="section-label")
                yield TextArea(
                    id="message-input",
                    placeholder="在这里输入需求、待优化的提示词或补充，中英文均可。",
                )
                with Horizontal(classes="button-row"):
                    yield Button("发送 F2", id="send", variant="primary")
                    yield Button("取消", id="cancel-run", variant="error")
                    yield Button("新会话", id="new-session")
            with Vertical(id="sidebar"):
                with TabbedContent():
                    with TabPane("设置", id="settings-tab"):
                        with VerticalScroll(classes="settings-pane"):
                            yield Button("配置模型与参数", id="configure-models", variant="primary")
                            yield Static("持久默认值在配置程序中修改；下方为本次会话设置。", id="configuration-note")
                            yield Label("本次会话 token 上限（可选）")
                            yield Input(placeholder="留空不设上限；填写正整数", id="token-budget")
                            yield Static("A/B 生成、C 独立评审。\n请求数和耗时仅作统计。\ntoken 预算可选，可随时取消。\n等待回答暂停计时；继续和切换保留消耗。", id="budget-note")
                            yield Static("本产品只交付提示词。软件工程任务留给后续执行。", id="product-note")
                    with TabPane("材料", id="materials-tab"):
                        with VerticalScroll(classes="settings-pane"):
                            with Vertical(id="material-list-box"):
                                yield Static("暂无材料。\n请在下方输入文件路径并点击“添加”。", id="material-list-hint")
                                yield OptionList(id="material-list")
                            yield Input(placeholder="在这里输入文件路径", id="material-path")
                            yield Label("参考用途")
                            yield Input(DEFAULT_PURPOSE, id="material-purpose")
                            yield Label("范围与遵循方式")
                            yield Input(DEFAULT_USAGE, id="material-usage")
                            yield Label("本次材料覆盖范围")
                            yield Select([("完整可提取文本", "full"), ("相关片段（未覆盖全文）", "relevant")], value="full", allow_blank=False, id="reference-mode")
                            with Horizontal(classes="button-row"):
                                yield Button("添加", id="add-material")
                                yield Button("移除", id="remove-material")
                                yield Button("预览", id="preview-materials")
                            yield Static("仅明确添加的本地文件会被加载。普通视频、图片与 URL 引用保留为文本。", id="materials-note")
                    with TabPane("历史", id="history-tab"):
                        with VerticalScroll(classes="settings-pane"):
                            yield Label("历史有内容时自动交接，清空后普通生成")
                            yield Select([("粘贴旧记录", "paste"), ("TXT / Markdown 文件", "files")], value="paste", allow_blank=False, id="history-source")
                            yield Static("粘贴完整旧记录；本轮目标写在左侧输入区。", id="history-note")
                            yield TextArea(id="history-input")
        yield Static("等待输入 · 请求 0 · 已知 token 0", id="status")
        yield Footer()

    def on_mount(self):
        self._resize_layout(self.size.width, self.size.height)
        self._log("系统", "输入需求即可开始。需要确认的关键缺口会在同一对话中提出。")
        self.query_one("#message-input", TextArea).focus()
        self._status("idle")
        self._update_controls()
        self._refresh_defaults()
        self._status_timer = self.set_interval(1, self._refresh_status, pause=True)

    def _resize_layout(self, width, height):
        screen = self.screen_stack[0]
        screen.set_class(width < 100, "narrow")
        screen.set_class(height <= 28, "compact")

    def on_resize(self, event):
        if self.screen_stack:
            self._resize_layout(event.size.width, event.size.height)

    def action_toggle_sidebar(self):
        if isinstance(self.screen, ModalScreen):
            return
        screen = self.screen_stack[0]
        screen.toggle_class("sidebar-open")

    def _log(self, role, text):
        self.query_one("#chat-log", RichLog).write(f"{role}  {text}")

    def _protected_paths(self):
        paths = [self.root / ".env", self.root / "context_windows.json", self.root / "optimizer_settings.json",
                 *[material.path for material in self.materials]]
        if self.query_one("#history-source", Select).value == "files":
            paths += [Path(line.strip().strip('"')).expanduser() for line in self.query_one("#history-input", TextArea).text.splitlines() if line.strip()]
        return tuple(paths)

    def _is_current(self, result):
        current = self._capture_inputs()
        return bool(self.controller is not None and self.controller.is_current(result)
                    and self.material_version == self._submitted_material_version
                    and current.reference_signature == self._submitted_reference_signature
                    and current.history_signature == self._submitted_history_signature
                    and not self.query_one("#message-input", TextArea).text.strip())

    def _update_controls(self):
        if not self.is_running:
            return
        ready = self.result is not None and self.result.status in {"ready", "unreviewed"} and self._is_current(self.result)
        for identifier in ("copy-prompt", "save-prompt"):
            self.query_one(f"#{identifier}", Button).disabled = self.busy or not ready
        self.query_one("#save-draft", Button).disabled = self.busy or not bool(self.query_one("#prompt-output", TextArea).text.strip())
        send = self.query_one("#send", Button)
        send.disabled = self.busy
        send.label = "继续对话 F2" if self.paused else "发送 F2"
        self.query_one("#cancel-run", Button).disabled = not self.busy or self._cancel_requested
        self.query_one("#new-session", Button).disabled = self.busy
        self.query_one("#token-budget", Input).disabled = self.busy or self.controller is not None
        self.query_one("#configure-models", Button).disabled = self.busy
        for identifier in ("material-path", "material-purpose", "material-usage", "reference-mode", "history-source", "history-input", "add-material", "remove-material", "preview-materials"):
            self.query_one(f"#{identifier}").disabled = self.busy
        self.query_one("#material-list", OptionList).display = bool(self.materials)
        self.query_one("#material-list-hint", Static).display = not self.materials

    def _set_busy(self, busy):
        self.busy = busy
        if self._status_timer is not None:
            self._status_timer.resume() if busy else self._status_timer.pause()
        self._update_controls()

    def _refresh_status(self):
        if (self.busy and self._core_active and not self.paused
                and self._last_phase not in {"questions", "paused"}):
            self._status()

    def _status(self, phase=None, metadata=None):
        if phase:
            self._last_phase = phase
        if metadata is None and self.controller is not None:
            metadata = self.controller.snapshot()
            optimizer = getattr(self.controller, "optimizer", None)
            if hasattr(optimizer, "activity_snapshot"):
                metadata = {**metadata, **optimizer.activity_snapshot()}
            elif hasattr(optimizer, "metadata"):
                metadata = {**metadata, **optimizer.metadata()}
        metadata = metadata or {}
        usage = metadata.get("usage", metadata.get("budget", metadata))
        count = usage.get("request_count", metadata.get("request_count", 0))
        known = usage.get("known_total_tokens", metadata.get("known_total_tokens", 0))
        unknown = usage.get("unknown_usage_requests", metadata.get("unknown_usage_requests", 0))
        workflow = metadata.get("workflow", self._capture_inputs().workflow if self.controller is None else "normal")
        elapsed = metadata.get("elapsed_seconds", 0)
        label = PHASE_LABELS.get(self._last_phase, STATUS_LABELS.get(self._last_phase, self._last_phase))
        if self._last_phase.startswith(("dialogue-clarification", "dialogue_clarification")):
            label = "判断关键缺口"
        elif self._last_phase.startswith("generate"):
            label = "生成提示词"
        elif self._last_phase.startswith(("review", "repair_review")):
            label = "独立评审"
        elif self._last_phase.startswith("repair"):
            label = "修复提示词"
        elif self._last_phase.startswith(("extract-", "merge-")):
            label = "整理与核验旧记录"
        suffix = f" · {unknown} 次用量未知" if unknown else ""
        flow_label = "历史交接" if workflow == "handoff" else "普通生成"
        summary = f"{flow_label} · {label} · 请求 {count} 次 · 耗时 {elapsed:g}s · 已知 token {known}{suffix}"
        active = metadata.get("active_calls") or []
        active = [call for call in active
                  if (call.get("role"), call.get("request_id")) not in self._finished_activity_requests
                  and call.get("request_id", 0) >= self._activity_request_ids.get(call.get("role"), 0)]
        if active:
            compact = self.size.width < 100 or self.size.height <= 28
            labels = SHORT_ACTIVITY_LABELS if compact else ACTIVITY_LABELS
            details = []
            for call in sorted(active, key=lambda row: {"a": 0, "b": 1, "judge": 2}.get(row.get("role"), 3)):
                role = {"a": "A", "b": "B", "judge": "C"}.get(call.get("role"), "模型")
                state = labels.get(call.get("activity_state"), "等待")
                if call.get("slow_waiting"):
                    state = "等待较久"
                timing = "" if compact else f"（{call.get('idle_seconds', 0):g}s 无新活动）"
                details.append(f"{role} {state}{timing}")
            detail = " / ".join(details)
            summary = (f"{label} · {detail} · {elapsed:g}s" if compact else summary + "\n" + detail)
        self.query_one("#status", Static).update(summary)

    @on(Select.Changed, "#history-source")
    def history_source_changed(self, event):
        self.query_one("#history-note", Static).update("每行一个 UTF-8 TXT / Markdown 路径，按顺序处理。" if event.value == "files" else "粘贴完整旧记录；本轮目标写在左侧输入区。")
        if self.controller is not None and not self.busy:
            self.material_version += 1
        self._update_controls()

    @on(Select.Changed, "#reference-mode")
    def reference_mode_changed(self, event):
        if self.controller is not None and not self.busy:
            self.material_version += 1
        self._update_controls()

    @on(TextArea.Changed, "#message-input")
    def input_changed(self, event):
        self._update_controls()

    @on(TextArea.Changed, "#history-input")
    def history_changed(self, event):
        if self.controller is not None and not self.busy:
            self.material_version += 1
        self._update_controls()

    @on(Button.Pressed)
    def pressed(self, event):
        actions = {
            "send": self.action_send, "cancel-run": self.action_cancel_run,
            "new-session": self.action_new_session,
            "copy-prompt": self.action_copy, "save-prompt": self.action_save_prompt,
            "save-draft": self.action_save_draft, "view-report": self.action_reports,
            "add-material": self.action_add_material, "remove-material": self.action_remove_material,
            "preview-materials": self.action_preview,
            "configure-models": self.action_configure,
        }
        action = actions.get(event.button.id)
        if action is not None:
            event.stop()
            action()

    def _refresh_defaults(self):
        try:
            settings = read_settings(self.root)
            self._runtime_settings = settings
            budget = settings["run"]["token_budget"]
            self.query_one("#token-budget", Input).value = str(budget) if budget is not None else ""
            self.query_one("#reference-mode", Select).value = settings["reference"]["mode"]
        except ConfigurationError as error:
            self.notify(str(error), severity="error")

    def _config_file_stamp(self):
        # Detect edits from the standalone program / CLI without reading credentials.
        result = []
        for name in (".env", "optimizer_settings.json", "context_windows.json"):
            try:
                metadata = (self.root / name).stat()
                result.append((metadata.st_mtime_ns, metadata.st_size))
            except OSError:
                result.append(None)
        return tuple(result)

    def action_configure(self):
        if self.busy or isinstance(self.screen, ModalScreen):
            return
        self.run_worker(self._configuration_worker(), name="配置", group="configuration", exit_on_error=False)

    async def _configuration_worker(self):
        from optimizer_config_ui import ConfigurationScreen
        from optimizer_settings import ConfigurationStore
        try:
            store = await asyncio.to_thread(self.configuration_store_factory or ConfigurationStore, self.root, recover=True)
            result = await self.push_screen_wait(ConfigurationScreen(store))
            if result is not None and result["changed"]:
                self._pending_configuration = self.controller is not None
                self.query_one("#configuration-note", Static).update(
                    "配置已保存。请新建会话生效；未提交需求、材料和历史将保留。" if self._pending_configuration
                    else "配置已保存，下次发送使用新参数。")
                if self.controller is None:
                    self._refresh_defaults()
                self._log("配置", "配置已保存，请重新发送；已有会话参数从新会话生效。")
        except (ConfigurationError, OSError, UnicodeError) as error:
            self.notify(str(error) if isinstance(error, ConfigurationError) else "配置无法读取。", severity="error")

    def action_add_material(self):
        if self.busy:
            return
        try:
            value = self.query_one("#material-path", Input).value.strip().strip('"')
            if not value:
                raise ConfigurationError("请填写材料路径。")
            path = check_reference_path(Path(value))
            if any(material.path == path or material.path.samefile(path) for material in self.materials):
                raise ConfigurationError("这份材料已经添加。")
            material = ReferenceFile(path, self.query_one("#material-purpose", Input).value, self.query_one("#material-usage", Input).value)
        except (ConfigurationError, OSError) as error:
            self.notify(str(error), severity="error")
            return
        self.materials.append(material)
        self.material_version += 1
        self.query_one("#material-list", OptionList).add_option(path.name)
        self.query_one("#material-path", Input).value = ""
        self._log("材料", f"{path.name} · 用途：{material.purpose} · 范围：{material.usage}")
        self._update_controls()

    def action_remove_material(self):
        if self.busy:
            return
        options = self.query_one("#material-list", OptionList)
        if options.highlighted is not None:
            index = options.highlighted
            material = self.materials.pop(index)
            options.remove_option_at_index(index)
            self.material_version += 1
            self._log("材料", "已移除 " + material.path.name)
            self._update_controls()

    def action_preview(self):
        if self.busy or not self.materials:
            return
        self._set_busy(True)
        self._preview_active = True
        self._cancel_requested = False
        self._worker = self.run_worker(self._preview_worker(), name="材料预览", group="interaction", exit_on_error=False)

    async def _preview_worker(self):
        try:
            self._status("preview")
            settings = read_settings(self.root) if self.controller is None else self._runtime_settings
            options = reference_options(settings, mode=self.query_one("#reference-mode", Select).value)
            query = self.query_one("#message-input", TextArea).text or self._submitted_text
            bundle = await asyncio.to_thread(prepare_references, tuple(self.materials), options, query=query)
            if not self._cancel_requested:
                await self.push_screen(PreviewScreen("\n\n".join([*bundle.blocks, *bundle.warnings])))
        except (OptimizerError, OSError, UnicodeError) as error:
            self.notify(str(error), severity="error")
        finally:
            self._preview_active = False
            self._cancel_requested = False
            self._set_busy(False)
            self._status("idle")

    def action_send(self):
        if isinstance(self.screen, QuestionScreen):
            self.screen.action_answer()
            return
        if self.busy or isinstance(self.screen, ModalScreen):
            return
        submission = self._capture_inputs()
        text = submission.text
        if not text.strip():
            if self.controller is None:
                self.notify("请输入需求或补充。")
                return
            inputs_unchanged = (self.material_version == self._submitted_material_version
                and submission.reference_signature == self._submitted_reference_signature
                and submission.history_signature == self._submitted_history_signature)
            if self.paused and inputs_unchanged:
                self.paused = False
                self._cancel_requested = False
                self._set_busy(True)
                self._worker = self.run_worker(self._resume_worker(), name="恢复对话",
                                               group="interaction", exit_on_error=False)
                return
            if inputs_unchanged:
                self.notify("请输入需求或补充。")
                return
            # Re-evaluate changed materials/history without inventing a user update.
            text = self.controller.snapshot().get("confirmed_request", "")
            if not text.strip():
                self.notify("请输入需求或补充。")
                return
            submission = replace(submission, text=text)
            self._log("系统", "材料或历史已变化，重新判断当前需求。")
        else:
            self._log("用户", text)
        self._submitted_text = text
        self._cancel_requested = False
        self.paused = False
        self._set_busy(True)
        self.query_one("#message-input", TextArea).load_text("")
        self._worker = self.run_worker(self._submit_worker(submission), name="优化会话", group="interaction", exit_on_error=False)

    def _capture_inputs(self):
        return _Submission(self.query_one("#message-input", TextArea).text,
                           tuple(self.materials), self.query_one("#reference-mode", Select).value,
                           self.query_one("#history-source", Select).value,
                           self.query_one("#history-input", TextArea).text, self.material_version,
                           self.query_one("#token-budget", Input).value.strip())

    async def _prepare_inputs(self, submission):
        # Reuse the complete cached chunks; the engine reselects against all
        # confirmed requirements rather than just this turn's short supplement.
        references = UNSET
        if self.controller is None or submission.reference_signature != self._submitted_reference_signature:
            options = reference_options(self._runtime_settings, mode=submission.reference_mode)
            confirmed = self.controller.snapshot().get("confirmed_request", "") if self.controller is not None else ""
            query = "\n\n".join(part for part in (confirmed, submission.text if submission.text != confirmed else "") if part)
            references = await asyncio.to_thread(prepare_references, submission.materials, options, query=query) if submission.materials else None
        history = UNSET
        if self.controller is None or submission.history_signature != self._submitted_history_signature:
            history = None
        if submission.workflow == "handoff" and history is None:
            if submission.history_source == "files":
                files = [Path(line.strip().strip('"')).expanduser() for line in submission.history_text.splitlines() if line.strip()]
                history = await asyncio.to_thread(prepare_history, files, options=history_options(self._runtime_settings))
            else:
                history = await asyncio.to_thread(prepare_history, text=submission.history_text, options=history_options(self._runtime_settings))
        if self._cancel_requested:
            raise _PreparationCancelled
        return references, history

    async def _ensure_windows(self, configs):
        path = self.root / "context_windows.json"
        try:
            existing = await asyncio.to_thread(read_window_config, path)
            WindowLimits.from_config(existing, configs)
            return existing
        except ConfigurationError as error:
            raise _MissingWindows(str(error) + " 请在设置中打开“配置模型与参数”，配置完成后重新发送；已有会话请先新建会话。") from None

    async def _submit_worker(self, submission):
        previous = (self._submitted_material_version, self._submitted_reference_signature,
                    self._submitted_history_signature)
        try:
            self._status("preparing")
            if self.controller is None:
                self._runtime_settings = read_settings(self.root)
            budget_text = submission.token_budget_text
            try:
                token_budget = int(budget_text) if budget_text else None
            except ValueError:
                raise ConfigurationError("会话 token 上限须为正整数，或留空不设上限。") from None
            options = run_options(self._runtime_settings, token_budget=token_budget)
            references, history = await self._prepare_inputs(submission)
            if self.controller is None:
                configs = await asyncio.to_thread(self.config_loader, self.root)
            else:
                configs = self.controller.optimizer.configs
            if self._cancel_requested:
                raise _PreparationCancelled
            windows = (await self._ensure_windows(configs) if history is not UNSET else UNSET) if submission.workflow == "handoff" else None
            if self._cancel_requested:
                raise _PreparationCancelled
            if self.controller is None:
                optimizer = self.optimizer_factory(configs, options)
                self.controller = self.controller_factory(optimizer, on_questions=self._on_questions, on_event=self._on_event)
                self._configuration_stamp = self._config_file_stamp()
            self._submitted_material_version = submission.material_version
            self._submitted_reference_signature = submission.reference_signature
            self._submitted_history_signature = submission.history_signature
            self._core_active = True
            result = await self.controller.submit(submission.text, references=references, history=history, context_windows=windows)
            self._core_active = False
            self._observe_result(result)
            await self._persist_result(result)
        except _PreparationCancelled:
            self._core_active = False
            self._submitted_material_version, self._submitted_reference_signature, self._submitted_history_signature = previous
            await self._preparation_failure("cancelled", "输入已取消；已确认内容和已有草稿保留在报告中。")
        except asyncio.CancelledError:
            self._core_active = False
            self._submitted_material_version, self._submitted_reference_signature, self._submitted_history_signature = previous
            if not self._quit_requested:
                await self._preparation_failure("cancelled", "操作已取消。")
        except (OptimizerError, OSError, UnicodeError) as error:
            self._core_active = False
            self._submitted_material_version, self._submitted_reference_signature, self._submitted_history_signature = previous
            await self._preparation_failure("handoff_configuration_error" if isinstance(error, _MissingWindows) else "failed", str(error))
        except Exception:
            self._core_active = False
            self._submitted_material_version, self._submitted_reference_signature, self._submitted_history_signature = previous
            await self._preparation_failure("failed", "运行发生未预期错误；服务详情未写入报告。")
        finally:
            self._core_active = False
            self._cancel_requested = False
            self._set_busy(False)
            self._status("paused" if self.paused else "finished", self.result.metadata if self.result else None)

    async def _preparation_failure(self, status, reason):
        if self.controller is not None:
            optimizer = getattr(self.controller, "optimizer", None)
            metadata = optimizer.metadata() if hasattr(optimizer, "metadata") else {}
            metadata = {**metadata, "dialogue": self.controller.snapshot()}
        else:
            metadata = {"request_count": 0, "known_total_tokens": 0, "total_tokens": 0,
                        "dialogue": {"session_id": self._provisional_session_id, "revision": 1,
                                     "original_request": self._submitted_text, "confirmed_request": self._submitted_text}}
        result = OptimizationResult(status, self.query_one("#prompt-output", TextArea).text or None, False, reason=reason, metadata=metadata)
        self._observe_result(result)
        await self._persist_result(result)
        if self.controller is None and not self.query_one("#message-input", TextArea).text:
            self.query_one("#message-input", TextArea).load_text(self._submitted_text)

    async def _on_questions(self, batch):
        if self._status_timer is not None:
            self._status_timer.pause()
        self._status("questions")
        for question in _value(batch, "questions", []):
            self._log("需要确认", _value(question, "text", ""))
        response = await self.push_screen_wait(QuestionScreen(batch))
        self.paused = response.get("action") == "pause"
        if response.get("action") == "answer" and self._status_timer is not None:
            self._status_timer.resume()
        for answer in response.get("answers", []):
            question = next((question for question in _value(batch, "questions", []) if _value(question, "id") == answer["question_id"]), None)
            selected = next((option for option in _value(question, "options", []) if _value(option, "id") == answer.get("option_id")), None)
            parts = [answer.get("text", "")]
            if selected is not None:
                parts.append("明确采用：" + str(_value(selected, "label", "")))
            self._log("用户回答", "\n".join(part for part in parts if part))
        return response

    def _on_event(self, event):
        if not self.is_running:
            return
        monitoring = event.get("kind") in {"activity", "stage", "usage"}
        if monitoring and event.get("session_id") is not None:
            if self.controller is None:
                return
            current = self.controller.snapshot()
            if any(event.get(key) != current.get(key) for key in ("session_id", "revision")):
                return
        if event.get("kind") == "activity":
            if not self.busy or self.paused or self._last_phase == "questions":
                return
        if monitoring:
            role, request_id = event.get("role"), event.get("request_id")
            if isinstance(request_id, int) and role:
                if (request_id < self._activity_request_ids.get(role, 0)
                        or (event.get("kind") == "activity"
                            and (role, request_id) in self._finished_activity_requests)):
                    return
                self._activity_request_ids[role] = request_id
                if event.get("kind") == "activity" and not any(
                        row.get("request_id") == request_id for row in event.get("active_calls", [])):
                    self._finished_activity_requests.add((role, request_id))
            if event.get("kind") == "stage" and self.busy and self._core_active and not self.paused:
                if self._status_timer is not None:
                    self._status_timer.resume()
        phase = event.get("phase") or event.get("kind")
        self._status(phase, event.get("metadata", event))
        if event.get("message"):
            self._log("状态", str(event["message"]))
        result = event.get("result")
        if isinstance(result, OptimizationResult):
            self._observe_result(result)
            worker = self.run_worker(self._persist_result(result), name="保存会话检查点", group="reports", exit_on_error=False)
            self._report_workers.append(worker)

    def _observe_result(self, result):
        if self.result is result:
            return
        if self._status_timer is not None:
            self._status_timer.pause()
        self.result = result
        self.query_one("#prompt-output", TextArea).load_text(result.optimized_prompt or "")
        self._log("结果", STATUS_LABELS.get(result.status, result.status) + (" · " + result.reason if result.reason else ""))
        if result.questions:
            for question in result.questions:
                self._log("未决项", str(question))
        dialogue = result.metadata.get("dialogue") or {}
        if dialogue.get("paused") or dialogue.get("state") == "paused" or dialogue.get("status") == "paused":
            self.paused = True
        self._status(metadata=result.metadata)
        self._update_controls()

    async def _persist_result(self, result, *, output_path=None, force=False):
        serialized = json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, default=str)
        key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        async with self._report_lock:
            if key in self._report_keys and not force:
                return
            try:
                outcome = await asyncio.to_thread(self.report_saver, result, root=self.root, current=None,
                                                 output_path=output_path, protected_paths=self._protected_paths())
                if _value(outcome, "report_saved", False):
                    self._report_keys.add(key)
                    path = _value(outcome, "report_path")
                    self.report_paths.append(path)
                    self._log("保存", "会话报告：" + str(path))
                for error in _value(outcome, "errors", []):
                    self.notify(str(error), severity="error")
            except Exception:
                self.notify("报告保存失败；界面中的内容仍可查看和另存。", severity="error")
            try:
                # Keep the final gate and atomic publication on the UI thread,
                # with no await where an edited draft could invalidate a result.
                outcome = self.prompt_saver(result, root=self.root, current=self._is_current,
                                            output_path=output_path, protected_paths=self._protected_paths())
                if _value(outcome, "prompt_saved", False):
                    self._log("保存", "提示词：" + str(_value(outcome, "prompt_path")))
                for error in _value(outcome, "errors", []):
                    self.notify(str(error), severity="error")
            except Exception:
                self.notify("提示词保存失败；界面中的结果仍可复制或重试。", severity="error")

    async def _resume_worker(self):
        try:
            if self._cancel_requested:
                raise _PreparationCancelled
            self._core_active = True
            result = await self.controller.resume()
            self._core_active = False
            self._observe_result(result)
            await self._persist_result(result)
        except _PreparationCancelled:
            self._core_active = False
            await self._preparation_failure("cancelled", "继续已取消；已确认内容和已有草稿保留在报告中。")
        except asyncio.CancelledError:
            self._core_active = False
            pass
        except Exception:
            self._core_active = False
            await self._preparation_failure("failed", "继续会话失败；已确认内容保留在报告中。")
        finally:
            self._core_active = False
            self._cancel_requested = False
            self._set_busy(False)
            self._status("paused" if self.paused else "finished", self.result.metadata if self.result else None)

    def action_cancel_run(self):
        if not self.busy or self._cancel_requested:
            return
        self._cancel_requested = True
        self._update_controls()
        self.run_worker(self._cancel_worker(), name="取消会话", group="cancel", exit_on_error=False)

    async def _cancel_worker(self):
        if self._preview_active:
            return
        if isinstance(self.screen, QuestionScreen):
            self.screen.dismiss({"action": "cancel", "answers": []})
        if self.controller is not None and self._core_active:
            result = await self.controller.cancel()
            if result is not None:
                self._observe_result(result)
                await self._persist_result(result)

    def action_new_session(self):
        if self.busy or isinstance(self.screen, ModalScreen):
            return
        self._set_busy(True)
        self.run_worker(self._new_session_worker(), name="新会话", group="interaction", exit_on_error=False)

    async def _new_session_worker(self):
        self._activity_request_ids.clear()
        self._finished_activity_requests.clear()
        try:
            preserve_inputs = self._pending_configuration or (self.controller is not None
                and self._configuration_stamp != self._config_file_stamp())
            pending_text = self.query_one("#message-input", TextArea).text if preserve_inputs else ""
            if self.controller is not None:
                await self.controller.close()
            self.controller = None
            self.result = None
            self.paused = False
            self._report_keys.clear()
            self._provisional_session_id = str(uuid4())
            self._submitted_text = ""
            self._submitted_reference_signature = None
            self._submitted_history_signature = None
            self.query_one("#chat-log", RichLog).clear()
            if not preserve_inputs:
                self.query_one("#prompt-output", TextArea).load_text("")
            self.query_one("#message-input", TextArea).load_text(pending_text)
            self._pending_configuration = False
            self._configuration_stamp = None
            self._refresh_defaults()
            self.query_one("#configuration-note", Static).update("持久默认值已加载；下方为本次会话设置。")
            self._log("系统", "已开始新会话。调用记录和用量已重置，已加载保存的默认参数。材料与历史保留供选择。" +
                      ("未提交需求和上次结果保留；上次结果仅供回看。" if preserve_inputs else "请输入本轮需求。"))
            self._status("idle", {})
        finally:
            self._set_busy(False)
            self.query_one("#message-input", TextArea).focus()

    def action_copy(self):
        if self.result is None or self.busy or self.result.status not in {"ready", "unreviewed"} or not self._is_current(self.result):
            self.notify("当前提示词已失效或仍待确认，请先完成本轮需求。")
            return
        self.run_worker(self._copy_worker(self.result), name="复制提示词", group="copy", exit_on_error=False)

    async def _copy_worker(self, result):
        if not self._is_current(result):
            return
        try:
            if self.clipboard_writer is not None:
                await asyncio.to_thread(self.clipboard_writer, result.optimized_prompt)
            elif sys.platform == "win32":
                await asyncio.to_thread(copy_windows_text, result.optimized_prompt)
            else:
                self.copy_to_clipboard(result.optimized_prompt)
            self.notify("提示词已复制。")
        except Exception as error:
            message = str(error) if isinstance(error, RuntimeError) else "复制失败，请重试或保存到文件。"
            self.notify(message, severity="error")

    def action_save_prompt(self):
        if self.result is None or self.busy or self.result.status not in {"ready", "unreviewed"} or not self._is_current(self.result):
            self.notify("当前提示词已失效或仍待确认。")
            return
        self.run_worker(self._save_prompt_worker(self.result), name="保存提示词", group="save", exit_on_error=False)

    async def _save_prompt_worker(self, result):
        path = await self.push_screen_wait(PathScreen("保存当前有效提示词", self.root / "optimized_prompt.md"))
        if path is not None:
            if not self._is_current(result):
                self.notify("需求已变化，旧提示词不能保存为当前结果。", severity="warning")
                return
            await self._persist_result(result, output_path=path, force=True)

    def action_save_draft(self):
        if self.busy:
            return
        text = self.query_one("#prompt-output", TextArea).text
        if text.strip():
            self.run_worker(self._save_draft_worker(text), name="另存草稿", group="save", exit_on_error=False)

    async def _save_draft_worker(self, text):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = await self.push_screen_wait(PathScreen("另存草稿（不替换上次成功提示词）", self.root / "drafts" / f"draft-{stamp}.md"))
        if path is not None:
            try:
                protected = (*self._protected_paths(), self.root / "last_optimized_prompt.md")
                saved = await asyncio.to_thread(self.draft_saver, text, path, protected_paths=protected)
                self._log("保存", "草稿：" + str(saved))
            except (OptimizerError, OSError, UnicodeError) as error:
                self.notify(str(error), severity="error")

    def action_reports(self):
        if isinstance(self.screen, ModalScreen):
            return
        self.run_worker(self._reports_worker(), name="离线回看报告", group="reports-view", exit_on_error=False)

    async def _reports_worker(self):
        path = await self.push_screen_wait(ReportPickerScreen(self.root))
        if path is None:
            return
        try:
            report = await asyncio.to_thread(self.report_loader, path)
            await self.push_screen(ReportScreen(path, report))
        except (OptimizerError, OSError, UnicodeError, ValueError):
            self.notify("报告无法读取或格式不完整。", severity="error")

    async def action_quit(self):
        if self._quit_requested:
            return
        self._quit_requested = True
        self._cancel_requested = True
        if isinstance(self.screen, QuestionScreen):
            self.screen.dismiss({"action": "cancel", "answers": []})
        if self.controller is not None:
            result = await self.controller.cancel() if self.busy else self.controller.result
            if result is not None:
                await self._persist_result(result)
            await self.controller.close()
        elif self.busy and not self._preview_active:
            await self._preparation_failure("cancelled", "已退出；本轮输入保留在会话报告中。")
        for worker in self._report_workers:
            try:
                await worker.wait()
            except Exception:
                pass
        self.exit()


def main(*, root=ROOT):
    OptimizerApp(root=root).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
