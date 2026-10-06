import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

from rich.cells import cell_len
from textual.widgets import Input, Select, Static, TextArea

from optimizer_config import ModelConfig
from optimizer_documents import LocalReferenceLoader, prepare_references
from optimizer_dialogue import UNSET
from optimizer_engine import OptimizationResult, Optimizer
from optimizer_handoff import required_roles
from tui import OptimizerApp, PathScreen, PreviewScreen, QuestionScreen, ReportPickerScreen, ReportScreen, WindowScreen, _publish_prompt, _save_result


class FakeController:
    def __init__(self, optimizer, *, on_questions, on_event, questions=False, gate=None):
        self.optimizer = optimizer
        self.on_questions, self.on_event = on_questions, on_event
        self.questions, self.gate = questions, gate
        self.submitted = []
        self.responses = []
        self.revision = 0
        self.result = None
        self.cancelled = False
        self.closed = False
        self.count = 0
        self.confirmed_request = ""
        self.workflow = "normal"
        self.limits = {"token_budget": optimizer.options.token_budget}

    def snapshot(self):
        return {"session_id": "fake-session", "revision": self.revision,
                "confirmed_request": self.confirmed_request, "workflow": self.workflow, "limits": dict(self.limits),
                "request_count": self.count, "known_total_tokens": self.count * 10, "unknown_usage_requests": 0}

    def is_current(self, result):
        return result.metadata.get("dialogue", {}).get("revision") == self.revision

    def finish(self, status="ready"):
        self.result = OptimizationResult(status, "最终中文提示词\n保留 {变量}" if status == "ready" else None, False,
                                         metadata={**self.snapshot(), "request_count": self.count, "known_total_tokens": self.count * 10,
                                                   "total_tokens": self.count * 10,
                                                   "dialogue": self.snapshot()})
        self.on_event({"kind": "result", "phase": "finished", "result": self.result})
        return self.result

    async def submit(self, text, **kwargs):
        self.submitted.append((text, kwargs))
        if text != self.confirmed_request:
            self.confirmed_request = "\n\n".join(part for part in (self.confirmed_request, text) if part)
        history = kwargs.get("history", UNSET)
        if history is not UNSET:
            self.workflow = "handoff" if history is not None else "normal"
        self.revision += 1
        self.count += 1
        self.on_event({"kind": "phase", "phase": "generation", "metadata": self.snapshot()})
        if self.questions:
            response = await self.on_questions({
                "session_id": "fake-session", "revision": self.revision,
                "round_number": 1, "paused": False,
                "questions": [
                    {"id": "permission", "text": "是否允许修改文件？", "reason": "这会改变操作权限。",
                     "options": [{"id": "read-only", "label": "只读检查"}, {"id": "edit", "label": "允许修改"}]},
                    {"id": "acceptance", "text": "验收要求是什么？", "reason": "这会改变验收方式。", "options": []},
                ],
            })
            self.responses.append(response)
            if response["action"] != "answer":
                return self.finish("cancelled" if response["action"] == "cancel" else "needs_clarification")
        if self.gate is not None:
            await self.gate.wait()
        return self.finish("cancelled" if self.cancelled else "ready")

    async def resume(self):
        self.count += 1
        return self.finish()

    async def cancel(self):
        self.cancelled = True
        if self.gate is not None:
            self.gate.set()
        return self.finish("cancelled")

    async def close(self):
        self.closed = True


class TuiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="optimizer tui 中文 ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.controllers = []
        self.loader = Mock(return_value={})

    def app(self, *, questions=False, gate=None, **kwargs):
        def factory(optimizer, **callbacks):
            controller = FakeController(optimizer, **callbacks, questions=questions, gate=gate)
            self.controllers.append(controller)
            return controller
        return OptimizerApp(root=self.root, config_loader=self.loader, controller_factory=factory, **kwargs)

    async def idle(self, app, pilot):
        for _ in range(80):
            await pilot.pause(0.01)
            if not app.busy:
                return
        self.fail("TUI did not return to idle")

    async def screen(self, app, pilot, expected):
        for _ in range(80):
            await pilot.pause(0.01)
            if isinstance(app.screen, expected):
                await pilot.pause()
                return
        self.fail(f"TUI did not open {expected.__name__}; actual {type(app.screen).__name__}")

    async def click(self, app, pilot, selector):
        await pilot.pause()
        widget = app.screen.query_one(selector)
        self.assertTrue(await pilot.click(selector, offset=(widget.size.width // 2, widget.size.height // 2)))
        await pilot.pause()

    async def test_boot_multiline_chinese_enter_and_f2_submit(self):
        app = self.app()
        async with app.run_test(size=(120, 38)) as pilot:
            self.loader.assert_not_called()
            for removed in ("workflow", "mode", "strict-review"):
                self.assertFalse(app.query(f"#{removed}"))
            note = str(app.query_one("#budget-note", Static).render())
            self.assertIn("请求数和耗时仅作统计", note)
            for retired_limit in ("12 次", "240 秒", "64 次", "900 秒"):
                self.assertNotIn(retired_limit, note)
            editor = app.query_one("#message-input", TextArea)
            editor.load_text("第一行中文 {字面值}")
            editor.move_cursor((0, len(editor.text)))
            editor.focus()
            await pilot.press("enter")
            editor.insert("第二行中文")
            await pilot.press("f2")
            await self.idle(app, pilot)
            self.assertEqual(self.controllers[0].submitted[0][0], "第一行中文 {字面值}\n第二行中文")
            self.assertFalse(hasattr(self.controllers[0].optimizer.options, "max_requests"))
            self.assertNotIn("max_requests", app.result.metadata["limits"])
            self.assertEqual(app.result.status, "ready")
            self.assertTrue((self.root / "last_optimized_prompt.md").is_file())
            self.assertTrue(app.report_paths)

    async def test_duplicate_submission_and_material_changes_are_locked(self):
        gate = asyncio.Event()
        app = self.app(gate=gate)
        async with app.run_test(size=(120, 38)) as pilot:
            app.query_one("#message-input", TextArea).load_text("第一次")
            app.action_send()
            await pilot.pause()
            app.query_one("#message-input", TextArea).load_text("运行时可以编辑")
            app.action_send()
            self.assertEqual(len(self.controllers[0].submitted), 1)
            self.assertTrue(app.query_one("#add-material").disabled)
            self.assertFalse(app.query_one("#message-input", TextArea).disabled)
            app.query_one("#message-input", TextArea).load_text("")
            gate.set()
            await self.idle(app, pilot)

    async def test_question_dialog_preserves_partial_free_answer_and_no_default_suggestion(self):
        app = self.app(questions=True)
        async with app.run_test(size=(120, 42)) as pilot:
            app.query_one("#message-input", TextArea).load_text("检查代码")
            app.action_send()
            await self.screen(app, pilot, QuestionScreen)
            self.assertIs(app.screen.query_one("#answer-option-0", Select).value, Select.NULL)
            await pilot.click("#answer-submit")
            self.assertIsInstance(app.screen, QuestionScreen)
            app.screen.query_one("#answer-text-1", TextArea).load_text("  保留中文原文\n输出 {JSON}  ")
            await pilot.press("f2")
            await self.idle(app, pilot)
            answers = self.controllers[0].responses[0]["answers"]
            self.assertEqual(answers, [{"question_id": "acceptance", "text": "  保留中文原文\n输出 {JSON}  ", "option_id": None}])

    async def test_question_explicit_option_and_pause_continue_share_controller(self):
        app = self.app(questions=True)
        async with app.run_test(size=(120, 42)) as pilot:
            app.query_one("#message-input", TextArea).load_text("检查代码")
            app.action_send()
            await self.screen(app, pilot, QuestionScreen)
            app.screen.query_one("#answer-option-0", Select).value = "read-only"
            await pilot.pause()
            await pilot.click("#answer-submit")
            await self.idle(app, pilot)
            self.assertEqual(self.controllers[0].responses[0]["answers"][0]["option_id"], "read-only")
            app.query_one("#message-input", TextArea).load_text("追加要求")
            app.action_send()
            await self.screen(app, pilot, QuestionScreen)
            await pilot.click("#answer-pause")
            await self.idle(app, pilot)
            self.assertTrue(app.paused)
            self.assertEqual(str(app.query_one("#send").label), "继续对话 F2")
            self.assertFalse(app.query("#continue-rounds"))
            count = self.controllers[0].count
            await pilot.press("f2")
            await self.idle(app, pilot)
            self.assertEqual(len(self.controllers), 1)
            self.assertEqual(self.controllers[0].count, count + 1)
            self.assertEqual(str(app.query_one("#send").label), "发送 F2")

    async def test_cancel_saves_report_and_preserves_previous_success(self):
        (self.root / "last_optimized_prompt.md").write_text("上次成功", encoding="utf-8")
        gate = asyncio.Event()
        app = self.app(gate=gate)
        async with app.run_test(size=(120, 38)) as pilot:
            app.query_one("#message-input", TextArea).load_text("本轮请求")
            app.action_send()
            await pilot.pause()
            app.action_cancel_run()
            await self.idle(app, pilot)
            self.assertTrue(self.controllers[0].cancelled)
            self.assertEqual(app.result.status, "cancelled")
            self.assertEqual((self.root / "last_optimized_prompt.md").read_text(encoding="utf-8"), "上次成功")
            reports = [json.loads(path.read_text(encoding="utf-8")) for path in (self.root / "reports").glob("*.json")]
            self.assertTrue(any(report["status"] == "cancelled" for report in reports))

    async def test_success_supplement_and_new_session_reset_only_explicitly(self):
        copied = []
        app = self.app(clipboard_writer=copied.append)
        async with app.run_test(size=(120, 38)) as pilot:
            editor = app.query_one("#message-input", TextArea)
            editor.load_text("原需求")
            app.action_send()
            await self.idle(app, pilot)
            app.action_copy()
            await pilot.pause()
            self.assertEqual(copied, [app.result.optimized_prompt])
            editor.load_text("补充验收条件")
            await pilot.pause()
            self.assertTrue(app.query_one("#copy-prompt").disabled)
            app.action_copy()
            await pilot.pause()
            self.assertEqual(len(copied), 1)
            app.action_send()
            await self.idle(app, pilot)
            self.assertEqual(len(self.controllers), 1)
            self.assertEqual(self.controllers[0].count, 2)
            app.action_new_session()
            await self.idle(app, pilot)
            self.assertTrue(self.controllers[0].closed)
            self.assertIsNone(app.controller)
            editor.load_text("新的独立需求")
            app.action_send()
            await self.idle(app, pilot)
            self.assertEqual(len(self.controllers), 2)
            self.assertEqual(self.controllers[1].count, 1)

    async def test_reference_preview_and_report_view_do_not_load_model_configuration(self):
        material = self.root / "reference.md"
        material_text = ("## 中文材料\n保留 {占位符} 与代码。\n"
                         + "访谈资料中的人物、时间与事件。" * 2000 + "\n最后一条访谈资料。\n")
        material.write_text(material_text, encoding="utf-8")
        report = self.root / "old-report.json"
        report.write_text(json.dumps({"status": "unreviewed", "optimized_prompt": "旧提示词"}, ensure_ascii=False), encoding="utf-8")
        app = self.app()
        async with app.run_test(size=(120, 38)) as pilot:
            app.query_one("#material-path", Input).value = str(material)
            app.action_add_material()
            app.action_preview()
            await self.screen(app, pilot, PreviewScreen)
            preview = app.screen.query_one("#preview-text", TextArea).text
            self.assertIn("中文材料", preview)
            self.assertIn("覆盖范围", preview)
            self.assertIn(str(material), preview)
            self.assertGreater(len(material_text), 20000)
            self.assertIn("最后一条访谈资料。", preview)
            self.loader.assert_not_called()
            await pilot.click("#preview-close")
            app.action_reports()
            await self.screen(app, pilot, ReportPickerScreen)
            app.screen.query_one("#report-path", Input).value = str(report)
            await pilot.click("#report-open")
            await self.screen(app, pilot, ReportScreen)
            self.assertIn("旧提示词", app.screen.query_one("#report-text", TextArea).text)
            self.assertIn("没有会话记录", app.screen.query_one("#report-dialogue", TextArea).text)
            self.loader.assert_not_called()

    async def test_resize_preserves_chinese_draft_and_can_submit_with_f2(self):
        app = self.app()
        async with app.run_test(size=(120, 38)) as pilot:
            app.query_one("#message-input", TextArea).load_text("缩放前中文\n缩放后也保留")
            await pilot.resize_terminal(80, 26)
            await pilot.pause()
            self.assertEqual(app.query_one("#message-input", TextArea).text, "缩放前中文\n缩放后也保留")
            await pilot.press("f2")
            await self.idle(app, pilot)
            self.assertEqual(self.controllers[0].submitted[0][0], "缩放前中文\n缩放后也保留")

    async def test_preparation_failure_saves_request_without_changing_success(self):
        app = self.app()
        app.config_loader = Mock(side_effect=RuntimeError("credential-detail-must-not-appear"))
        (self.root / "last_optimized_prompt.md").write_text("上次成功", encoding="utf-8")
        async with app.run_test(size=(120, 38)) as pilot:
            app.query_one("#message-input", TextArea).load_text("应保留的原始需求")
            app.action_send()
            await self.idle(app, pilot)
            data = json.loads(app.report_paths[-1].read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "failed")
            self.assertEqual(data["metadata"]["dialogue"]["original_request"], "应保留的原始需求")
            self.assertNotIn("credential-detail-must-not-appear", json.dumps(data))
            self.assertEqual((self.root / "last_optimized_prompt.md").read_text(encoding="utf-8"), "上次成功")

    async def test_80_by_24_buttons_are_visible_and_clickable_before_and_after_resize(self):
        copied = []
        gate = asyncio.Event()
        app = self.app(gate=gate, clipboard_writer=copied.append)
        async with app.run_test(size=(80, 24)) as pilot:
            self.assertFalse(app.query("#continue-rounds"))
            app.query_one("#message-input", TextArea).load_text("窄终端请求")
            self.assertTrue(await pilot.click("#send", offset=(3, 1)))
            await pilot.pause()
            self.assertTrue(await pilot.click("#cancel-run", offset=(3, 1)))
            await self.idle(app, pilot)
            self.assertTrue(await pilot.click("#new-session", offset=(3, 1)))
            await self.idle(app, pilot)
            app.query_one("#message-input", TextArea).load_text("新请求")
            await pilot.press("f2")
            await self.idle(app, pilot)
            self.assertTrue(await pilot.click("#copy-prompt", offset=(3, 1)))
            await pilot.pause()
            self.assertEqual(len(copied), 1)
            await pilot.resize_terminal(120, 40)
            await pilot.pause()
            self.assertTrue(await pilot.click("#copy-prompt", offset=(3, 1)))
            await pilot.pause()
            self.assertEqual(len(copied), 2)

    async def test_token_budget_validation_retains_request_and_locks_valid_budget(self):
        app = self.app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#message-input", TextArea).load_text("保留原需求")
            app.query_one("#token-budget", Input).value = "-1"
            app.action_send()
            await self.idle(app, pilot)
            self.loader.assert_not_called()
            self.assertEqual(app.result.status, "failed")
            self.assertEqual(app.query_one("#message-input", TextArea).text, "保留原需求")
            app.query_one("#token-budget", Input).value = "1000"
            app.action_send()
            await self.idle(app, pilot)
            self.assertEqual(self.controllers[0].optimizer.options.token_budget, 1000)
            self.assertTrue(app.query_one("#token-budget", Input).disabled)

    async def test_usage_events_show_top_level_known_and_unknown_tokens(self):
        app = self.app()
        async with app.run_test(size=(120, 40)) as pilot:
            app._on_event({"kind": "usage", "phase": "generate", "request_count": 65,
                           "elapsed_seconds": 1000,
                           "limits": {"max_requests": 12, "total_timeout": 240},
                           "known_total_tokens": 120, "unknown_usage_requests": 1})
            text = str(app.query_one("#status", Static).render())
            self.assertIn("请求 65 次", text)
            self.assertIn("耗时 1000s", text)
            self.assertNotIn("/12", text)
            self.assertNotIn("/240", text)
            self.assertIn("已知 token 120", text)
            self.assertIn("1 次用量未知", text)

    async def test_activity_statuses_fit_wide_and_80_by_24_without_streaming_text(self):
        states = (("waiting", "等待响应", "等待"), ("thinking", "正在思考", "思考"),
                  ("receiving", "正在接收结果", "接收"), ("validating", "正在校验", "校验"),
                  ("retrying", "正在重试", "重试"))
        for size in ((120, 40), (80, 24)):
            with self.subTest(size=size):
                app = self.app()
                async with app.run_test(size=size) as pilot:
                    app.query_one("#prompt-output", TextArea).load_text("已完成的草稿")
                    for state, wide, narrow in states:
                        metadata = {"active_calls": [{"role": role, "request_id": index,
                                     "activity_state": state, "idle_seconds": 12}
                                     for index, role in enumerate(("a", "b", "judge"), 1)],
                                    "elapsed_seconds": 15, "known_total_tokens": 300, "request_count": 3}
                        app._status("generate", metadata)
                        await pilot.pause()
                        status = app.query_one("#status", Static)
                        content = str(status.content)
                        for role in ("A", "B", "C"):
                            self.assertIn(f"{role} {wide if size[0] > 100 else narrow}", content)
                        self.assertEqual(len(content.splitlines()), 2 if size[0] > 100 else 1)
                        for line in content.splitlines():
                            self.assertLessEqual(cell_len(line), status.content_size.width)
                        self.assertGreater(status.region.y, 0)
                        self.assertLess(status.region.bottom, size[1])
                        self.assertEqual(app.query_one("#prompt-output", TextArea).text, "已完成的草稿")
                    metadata["active_calls"][0]["slow_waiting"] = True
                    app._status("review", metadata)
                    self.assertIn("A 等待较久", str(app.query_one("#status", Static).content))
                    self.loader.assert_not_called()

    async def test_activity_events_ignore_stale_context_request_and_completion(self):
        gate = asyncio.Event()
        app = self.app(gate=gate)
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#message-input", TextArea).load_text("只看状态")
            app.action_send()
            await pilot.pause()
            controller = self.controllers[0]
            def event(request_id=3, state="thinking", **changes):
                return {"kind": "activity", "phase": "generate", "role": "a", "request_id": request_id,
                        **controller.snapshot(), "active_calls": [{"role": "a", "request_id": request_id,
                                                                 "activity_state": state}], **changes}
            app._on_event(event())
            original = str(app.query_one("#status", Static).content)
            self.assertIn("A 正在思考", original)
            for stale in (event(2, "receiving"), event(session_id="retired-session"), event(revision=0)):
                app._on_event(stale)
                self.assertEqual(str(app.query_one("#status", Static).content), original)
            app._on_event(event(active_calls=[]))
            completed = str(app.query_one("#status", Static).content)
            app._on_event(event())
            self.assertEqual(str(app.query_one("#status", Static).content), completed)
            app._on_event(event(4, "waiting"))
            self.assertIn("A 等待响应", str(app.query_one("#status", Static).content))
            app.paused = True
            app._on_event(event(5, "receiving"))
            self.assertIn("A 等待响应", str(app.query_one("#status", Static).content))
            app.paused = False
            app._status("questions", {})
            app._on_event(event(5, "receiving"))
            self.assertIn("等待回答", str(app.query_one("#status", Static).content))
            gate.set()
            await self.idle(app, pilot)
            finished = str(app.query_one("#status", Static).content)
            app._on_event(event(6, "thinking"))
            self.assertEqual(str(app.query_one("#status", Static).content), finished)
            app.action_new_session()
            await self.idle(app, pilot)
            self.assertEqual(app._activity_request_ids, {})
            self.assertFalse(app._finished_activity_requests)
            self.assertIn("等待输入", str(app.query_one("#status", Static).content))

    async def test_one_second_refresh_uses_live_snapshot_and_stops_while_idle_or_paused(self):
        gate = asyncio.Event()
        app = self.app(gate=gate)
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#message-input", TextArea).load_text("等待状态刷新")
            app.action_send()
            await pilot.pause()
            optimizer = self.controllers[0].optimizer
            activity = {"active_calls": [{"role": "judge", "request_id": 1, "activity_state": "waiting",
                                         "idle_seconds": 91, "slow_waiting": True}], "elapsed_seconds": 91}
            snapshot = Mock(side_effect=lambda: activity)
            optimizer.activity_snapshot = snapshot
            await pilot.pause(1.1)
            self.assertIn("C 等待较久", str(app.query_one("#status", Static).content))
            self.assertTrue(app.busy)
            activity["active_calls"][0].update(activity_state="receiving", idle_seconds=0, slow_waiting=False)
            await pilot.pause(1.1)
            self.assertIn("C 正在接收结果", str(app.query_one("#status", Static).content))
            for paused, phase in ((True, "paused"), (False, "questions")):
                app.paused = paused
                app._status(phase, {})
                before = snapshot.call_count
                app._refresh_status()
                self.assertEqual(snapshot.call_count, before)
            gate.set()
            await self.idle(app, pilot)
            before = snapshot.call_count
            await pilot.pause(1.1)
            self.assertEqual(snapshot.call_count, before)

    async def test_80_by_24_f3_opens_settings_and_preserves_draft_and_budget(self):
        app = self.app()
        async with app.run_test(size=(80, 24)) as pilot:
            editor = app.query_one("#message-input", TextArea)
            editor.load_text("窄屏中文需求\n保留 {原文}")
            self.assertFalse(app.query_one("#sidebar").display)
            await pilot.press("f3")
            await pilot.pause()
            self.assertTrue(app.query_one("#sidebar").display)
            self.assertFalse(app.query_one("#conversation").display)
            budget = app.query_one("#token-budget", Input)
            budget.focus()
            await pilot.press("1", "0", "0", "0")
            self.assertEqual(budget.value, "1000")
            await pilot.press("f3")
            await pilot.pause()
            self.assertTrue(app.query_one("#conversation").display)
            self.assertEqual(editor.text, "窄屏中文需求\n保留 {原文}")
            await self.click(app, pilot, "#send")
            await self.idle(app, pilot)
            self.assertEqual(self.controllers[0].optimizer.options.token_budget, 1000)
            self.assertTrue(budget.disabled)
            await pilot.resize_terminal(120, 40)
            await pilot.pause()
            self.assertTrue(app.query_one("#sidebar").display)
            self.assertEqual(budget.value, "1000")

    async def test_80_by_24_questions_submit_pause_and_cancel(self):
        for action in ("answer", "pause", "cancel"):
            with self.subTest(action=action):
                app = self.app(questions=True)
                async with app.run_test(size=(80, 24)) as pilot:
                    app.query_one("#message-input", TextArea).load_text("小窗口检查代码")
                    await self.click(app, pilot, "#send")
                    await self.screen(app, pilot, QuestionScreen)
                    if action == "answer":
                        await self.click(app, pilot, "#answer-submit")
                        self.assertIsInstance(app.screen, QuestionScreen)
                        self.assertIn("至少回答", str(app.screen.query_one("#question-error", Static).render()))
                        answer = app.screen.query_one("#answer-text-1", TextArea)
                        answer.focus()
                        answer.load_text("保留全部原始条件\n只提供提示词")
                        await pilot.press("f2")
                    else:
                        await self.click(app, pilot, f"#answer-{action}")
                    await self.idle(app, pilot)
                    controller = self.controllers[-1]
                    self.assertEqual(len(controller.submitted), 1)
                    self.assertEqual(controller.responses[0]["action"], action)
                    self.assertEqual(app.paused, action == "pause")
                    if action == "answer":
                        self.assertEqual(controller.responses[0]["answers"], [{
                            "question_id": "acceptance", "text": "保留全部原始条件\n只提供提示词", "option_id": None,
                        }])
                        self.assertEqual(app.result.status, "ready")
                    else:
                        self.assertEqual(app.result.status, "needs_clarification" if action == "pause" else "cancelled")

    async def test_80_by_24_quality_windows_scroll_validate_save_and_cancel(self):
        configs = {role: ModelConfig(role, f"offline-{role}", "https://example.invalid/v1", "dummy")
                   for role in required_roles()}
        responses = []
        app = self.app()
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(WindowScreen(configs), responses.append)
            await self.screen(app, pilot, WindowScreen)
            await self.click(app, pilot, "#windows-save")
            self.assertIsInstance(app.screen, WindowScreen)
            self.assertIn("正整数", str(app.screen.query_one("#window-error", Static).render()))
            for role in required_roles():
                window = app.screen.query_one(f"#window-{role}", Input)
                window.focus()
                await pilot.pause()
                await pilot.press("6", "5", "5", "3", "6")
                self.assertEqual(window.value, "65536")
            await self.click(app, pilot, "#windows-save")
            self.assertEqual(responses, [{role: 65536 for role in required_roles()}])
            await app.push_screen(WindowScreen(configs), responses.append)
            await self.screen(app, pilot, WindowScreen)
            await self.click(app, pilot, "#windows-cancel")
            self.assertEqual(responses[-1], None)
            self.loader.assert_not_called()
            self.assertFalse((self.root / "context_windows.json").exists())

    async def test_80_by_24_failed_report_and_prompt_save_can_retry(self):
        saver = Mock(side_effect=OSError("保存位置暂时不可写"))
        prompt_saver = Mock(side_effect=OSError("提示词保存位置暂时不可写"))
        app = self.app(report_saver=saver, prompt_saver=prompt_saver)
        app.notify = Mock()
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#message-input", TextArea).load_text("保存失败也保留结果")
            await self.click(app, pilot, "#send")
            await self.idle(app, pilot)
            await app.workers.wait_for_complete()
            result = app.result
            prompt = app.query_one("#prompt-output", TextArea).text
            self.assertEqual(app.report_paths, [])
            self.assertFalse(app.query_one("#save-prompt").disabled)
            app.notify.assert_any_call("报告保存失败；界面中的内容仍可查看和另存。", severity="error")
            await self.click(app, pilot, "#save-prompt")
            await self.screen(app, pilot, PathScreen)
            await self.click(app, pilot, "#path-save")
            await app.workers.wait_for_complete()
            self.assertFalse((self.root / "optimized_prompt.md").exists())
            self.assertIs(app.result, result)
            self.assertEqual(app.query_one("#prompt-output", TextArea).text, prompt)
            saver.side_effect = _save_result
            prompt_saver.side_effect = _publish_prompt
            await self.click(app, pilot, "#save-prompt")
            await self.screen(app, pilot, PathScreen)
            await self.click(app, pilot, "#path-save")
            await app.workers.wait_for_complete()
            self.assertEqual((self.root / "optimized_prompt.md").read_text(encoding="utf-8"), prompt)
            self.assertEqual(len(app.report_paths), 1)
            self.assertEqual(len(self.controllers[0].submitted), 1)

    async def test_failed_background_report_still_publishes_current_prompt(self):
        app = self.app(report_saver=Mock(side_effect=OSError("报告不可写")))
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#message-input", TextArea).load_text("报告失败也保留有效提示词")
            await self.click(app, pilot, "#send")
            await self.idle(app, pilot)
            await app.workers.wait_for_complete()
            self.assertEqual(app.report_paths, [])
            self.assertEqual((self.root / "last_optimized_prompt.md").read_text(encoding="utf-8"), app.result.optimized_prompt)

    async def test_edit_during_background_report_prevents_real_prompt_publication(self):
        loop = asyncio.get_running_loop()
        ui_thread = threading.get_ident()
        report_entered = asyncio.Event()
        release_report = threading.Event()
        report_threads, publication_threads = [], []
        previous = "上次成功；新的未提交条件不应被旧结果覆盖"
        (self.root / "last_optimized_prompt.md").write_text(previous, encoding="utf-8")

        def report_saver(result, **kwargs):
            self.assertIsNone(kwargs["current"])
            report_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(report_entered.set)
            if not release_report.wait(5):
                raise AssertionError("后台报告未及时释放")
            return _save_result(result, **kwargs)

        def prompt_saver(result, **kwargs):
            publication_threads.append(threading.get_ident())
            return _publish_prompt(result, **kwargs)

        app = self.app(report_saver=report_saver, prompt_saver=prompt_saver)
        async with app.run_test(size=(80, 24)) as pilot:
            try:
                editor = app.query_one("#message-input", TextArea)
                editor.load_text("本轮原需求")
                await self.click(app, pilot, "#send")
                await asyncio.wait_for(report_entered.wait(), timeout=5)
                editor.load_text("新的未提交验收条件")
                await pilot.pause()
                self.assertFalse(app._is_current(app.result))
                release_report.set()
                await self.idle(app, pilot)
                await app.workers.wait_for_complete()
                self.assertEqual(app.result.status, "ready")
                self.assertTrue(app.report_paths)
                self.assertEqual((self.root / "last_optimized_prompt.md").read_text(encoding="utf-8"), previous)
                self.assertTrue(report_threads)
                self.assertTrue(all(identifier != ui_thread for identifier in report_threads))
                self.assertTrue(publication_threads)
                self.assertEqual(set(publication_threads), {ui_thread})
            finally:
                release_report.set()

    async def test_handoff_supplement_reuses_unchanged_history_without_reloading(self):
        history = self.root / "旧记录.md"
        history.write_text("旧记录：必须只读检查，保留全部约束。", encoding="utf-8")
        app = self.app()
        app._ensure_windows = AsyncMock(return_value={})
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#history-source", Select).value = "files"
            app.query_one("#history-input", TextArea).load_text(str(history))
            app.query_one("#message-input", TextArea).load_text("根据旧记录检查代码")
            await pilot.press("f2")
            await self.idle(app, pilot)
            controller = self.controllers[0]
            self.assertIsNotNone(controller.submitted[0][1]["history"])
            self.assertIsNot(controller.submitted[0][1]["history"], UNSET)
            history.unlink()
            with patch("tui.prepare_history", side_effect=AssertionError("未改历史不得重读")):
                app.query_one("#message-input", TextArea).load_text("追加：回答简洁")
                await pilot.press("f2")
                await self.idle(app, pilot)
            self.assertEqual(app.result.status, "ready")
            self.assertEqual(len(self.controllers), 1)
            self.assertEqual(len(controller.submitted), 2)
            self.assertIs(controller.submitted[1][1]["history"], UNSET)
            app._ensure_windows.assert_awaited_once()

    async def test_whitespace_history_is_normal_for_both_sources(self):
        for source in ("paste", "files"):
            with self.subTest(source=source):
                app = self.app()
                app._ensure_windows = AsyncMock(side_effect=AssertionError("普通生成无需交接窗口"))
                async with app.run_test(size=(80, 24)) as pilot:
                    app.query_one("#history-source", Select).value = source
                    app.query_one("#history-input", TextArea).load_text(" \n\t ")
                    app.query_one("#message-input", TextArea).load_text("只检查，不改文件。")
                    with patch("tui.prepare_history", side_effect=AssertionError("空白历史不得加载")):
                        await pilot.press("f2")
                        await self.idle(app, pilot)
                    submitted = self.controllers[-1].submitted[0][1]
                    self.assertIsNone(submitted["history"])
                    self.assertIsNone(submitted["context_windows"])
                    app._ensure_windows.assert_not_awaited()

    async def test_history_only_send_reuses_confirmed_request_and_clear_is_explicit(self):
        app = self.app()
        app._ensure_windows = AsyncMock(return_value={})
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#message-input", TextArea).load_text("检查代码，保留 {变量}。")
            await pilot.press("f2")
            await self.idle(app, pilot)
            controller = self.controllers[-1]
            app.query_one("#history-input", TextArea).load_text("用户：只读检查。")
            await pilot.pause()
            self.assertTrue(app.query_one("#copy-prompt").disabled)
            await pilot.press("f2")
            await self.idle(app, pilot)
            self.assertEqual(len(self.controllers), 1)
            self.assertEqual(controller.submitted[1][0], "检查代码，保留 {变量}。")
            self.assertIsNotNone(controller.submitted[1][1]["history"])
            self.assertEqual(controller.revision, 2)
            app.query_one("#history-input", TextArea).load_text("")
            await pilot.pause()
            await pilot.press("f2")
            await self.idle(app, pilot)
            self.assertEqual(controller.submitted[2][0], "检查代码，保留 {变量}。")
            self.assertIsNone(controller.submitted[2][1]["history"])
            self.assertIsNone(controller.submitted[2][1]["context_windows"])
            app._ensure_windows.assert_awaited_once()
            await pilot.press("f2")
            await pilot.pause()
            self.assertEqual(len(controller.submitted), 3, "空输入且材料历史未变不重复生成")

    async def test_send_freezes_history_before_worker_and_rejects_changed_result(self):
        app = self.app()
        app._ensure_windows = AsyncMock(return_value={})
        last = self.root / "last_optimized_prompt.md"
        last.write_text("上次成功", encoding="utf-8")
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#message-input", TextArea).load_text("依据旧记录检查代码。")
            app.query_one("#history-input", TextArea).load_text("用户：发送时的旧记录。")
            with patch("tui.prepare_history", return_value=object()) as prepare:
                app.action_send()
                app.query_one("#history-input", TextArea).load_text("用户：发送后的新记录。")
                await self.idle(app, pilot)
            self.assertEqual(prepare.call_args.kwargs["text"], "用户：发送时的旧记录。")
            self.assertEqual(app._submitted_history_signature[-1], "用户：发送时的旧记录。")
            self.assertTrue(app.query_one("#copy-prompt").disabled)
            self.assertEqual(last.read_text(encoding="utf-8"), "上次成功")

    async def test_cancel_material_preparation_preserves_existing_controller_and_can_retry(self):
        material = self.root / "new-material.md"
        material.write_text("新材料：保持只读。", encoding="utf-8")
        app = self.app()
        entered, release = threading.Event(), threading.Event()
        def slow_prepare(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError("准备取消测试未释放文件读取")
            return prepare_references(*args, **kwargs)
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#message-input", TextArea).load_text("检查代码，不改文件。")
            await pilot.press("f2")
            await self.idle(app, pilot)
            controller = self.controllers[-1]
            before = controller.snapshot()
            last = self.root / "last_optimized_prompt.md"
            previous_prompt = last.read_text(encoding="utf-8")
            app.query_one("#material-path", Input).value = str(material)
            app.action_add_material()
            with patch("tui.prepare_references", side_effect=slow_prepare):
                app.action_send()
                for _ in range(80):
                    await pilot.pause(0.01)
                    if entered.is_set():
                        break
                self.assertTrue(entered.is_set())
                app.action_cancel_run()
                await pilot.pause()
                self.assertFalse(controller.cancelled)
                release.set()
                await self.idle(app, pilot)
            self.assertEqual(app.result.status, "cancelled")
            self.assertIs(app.controller, controller)
            self.assertEqual(controller.snapshot(), before)
            self.assertFalse(controller.closed)
            self.assertEqual(last.read_text(encoding="utf-8"), previous_prompt)
            await pilot.press("f2")
            await self.idle(app, pilot)
            self.assertEqual(app.result.status, "ready")
            self.assertIs(app.controller, controller)
            self.assertEqual(controller.count, 2)

    async def test_real_relevant_materials_reuse_cache_and_changed_sources_use_complete_request(self):
        from tests.test_dialogue import DialogueScripts

        scripts = DialogueScripts()
        configs = {role: ModelConfig(role, "offline-" + role, "https://example.invalid/v1", "dummy")
                   for role in required_roles()}
        app = OptimizerApp(root=self.root, config_loader=lambda root: configs,
                           optimizer_factory=lambda configs, options: Optimizer(configs, options, factory=scripts.factory))
        material = self.root / "原始材料.md"
        material.write_text("alpha 订单缓存方案。", encoding="utf-8")
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#reference-mode", Select).value = "relevant"
            app.query_one("#material-path", Input).value = str(material)
            app.action_add_material()
            app.query_one("#message-input", TextArea).load_text("检查 alpha 订单缓存方案，不修改文件。")
            await pilot.press("f2")
            await self.idle(app, pilot)
            self.assertEqual(app.result.status, "ready")
            original_hash = app.controller.optimizer.references.files[0].sha256
            with patch.object(LocalReferenceLoader, "load", side_effect=AssertionError("未改材料不得重读")), \
                    patch("optimizer_documents.check_reference_path", side_effect=AssertionError("缓存无需重新检查磁盘")):
                for supplement in ("追加：回复温柔克制。",):
                    app.query_one("#message-input", TextArea).load_text(supplement)
                    await pilot.press("f2")
                    await self.idle(app, pilot)
                    self.assertEqual(app.result.status, "ready")
                    self.assertIn(supplement, app.controller.snapshot()["confirmed_request"])
                    self.assertEqual(app.controller.optimizer.references.files[0].sha256, original_hash)
            added = self.root / "新增材料.md"
            added.write_text("alpha 补充验收方案。", encoding="utf-8")
            app.query_one("#material-path", Input).value = str(added)
            app.action_add_material()
            app.query_one("#message-input", TextArea).load_text("追加：也请给出建议。")
            with patch("tui.prepare_references", wraps=prepare_references) as prepare:
                await pilot.press("f2")
                await self.idle(app, pilot)
                prepare.assert_called_once()
                query = prepare.call_args.kwargs["query"]
                self.assertIn("检查 alpha 订单缓存方案", query)
                self.assertIn("追加：回复温柔克制。", query)
                self.assertIn("追加：也请给出建议。", query)
            self.assertEqual(app.result.status, "ready")
            self.assertEqual(len(app.controller.optimizer.references.files), 2)
            self.assertEqual(app.controller.snapshot()["original_request"], "检查 alpha 订单缓存方案，不修改文件。")

    async def test_copy_failure_keeps_prompt_and_can_retry_from_button(self):
        clipboard = Mock(side_effect=RuntimeError("剪贴板正在被其他程序使用，请稍后重试。"))
        app = self.app(clipboard_writer=clipboard)
        app.notify = Mock()
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#message-input", TextArea).load_text("复制失败可以重试")
            await self.click(app, pilot, "#send")
            await self.idle(app, pilot)
            await self.click(app, pilot, "#copy-prompt")
            await app.workers.wait_for_complete()
            app.notify.assert_any_call("剪贴板正在被其他程序使用，请稍后重试。", severity="error")
            self.assertFalse(app.query_one("#copy-prompt").disabled)
            prompt = app.query_one("#prompt-output", TextArea).text
            clipboard.side_effect = None
            await self.click(app, pilot, "#copy-prompt")
            await app.workers.wait_for_complete()
            self.assertEqual(clipboard.call_count, 2)
            clipboard.assert_called_with(prompt)
            app.notify.assert_any_call("提示词已复制。")

    async def test_80_by_24_report_picker_open_and_close_stay_accessible(self):
        report = self.root / "old-report.json"
        report.write_text(json.dumps({"status": "unreviewed", "optimized_prompt": "中文旧提示词"}, ensure_ascii=False), encoding="utf-8")
        app = self.app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.click(app, pilot, "#view-report")
            await self.screen(app, pilot, ReportPickerScreen)
            path = app.screen.query_one("#report-path", Input)
            path.focus()
            path.value = str(report)
            await self.click(app, pilot, "#report-open")
            await self.screen(app, pilot, ReportScreen)
            self.assertIn("中文旧提示词", app.screen.query_one("#report-text", TextArea).text)
            await self.click(app, pilot, "#report-close")
            self.loader.assert_not_called()

    async def test_real_terminal_probe_controller_ctrl_q_closes_after_save(self):
        from evals.tui_windows_probe import EXPECTED, make_app

        app, submitted = make_app(self.root)
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#message-input", TextArea).load_text(EXPECTED)
            await pilot.press("f2")
            await self.idle(app, pilot)
            self.assertTrue(app.report_paths)
            await asyncio.wait_for(pilot.press("ctrl+q"), timeout=5)
            await asyncio.wait_for(asyncio.shield(app._task), timeout=5)
            self.assertTrue(app._quit_requested)
            self.assertTrue(app.controller.snapshot()["closed"])
            self.assertEqual(submitted, [EXPECTED])

    async def test_real_terminal_probe_controller_timer_closes_after_save(self):
        from evals.tui_windows_probe import EXPECTED, make_app

        app, submitted = make_app(self.root, terminal=True)
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#message-input", TextArea).load_text(EXPECTED)
            await pilot.press("f2")
            await asyncio.wait_for(asyncio.shield(app._task), timeout=5)
            self.assertTrue(app._quit_requested)
            self.assertTrue(app.controller.snapshot()["closed"])
            self.assertTrue(app.report_paths)
            self.assertEqual(app.result.status, "ready")
            self.assertEqual(submitted, [EXPECTED])


if __name__ == "__main__":
    unittest.main()
