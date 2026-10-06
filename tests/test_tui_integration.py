"""Cross-module TUI tests with the real controller/engine and offline models."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from optimizer_engine import Optimizer
from optimizer_models import ModelActivity
from optimizer_handoff import model_identity, required_roles
from test_dialogue import DialogueScripts, ORIGINAL, ask, draft, question, sufficient
from test_dialogue_handoff import ScriptedCalls, models

try:
    from textual.widgets import Input, Select, Static, TextArea
    from tui import OptimizerApp, QuestionScreen, WindowScreen
except ImportError:
    OptimizerApp = None


@unittest.skipIf(OptimizerApp is None, "optional Textual dependency is not installed")
class TuiIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.configs = models()
        self.engines = []

    def app(self, scripts, *, history_calls=None):
        base_factory = scripts.factory
        def factory(config):
            normal = base_factory(config)
            if history_calls is None:
                return normal
            class Model:
                def disable_streaming(self):
                    normal.disable_streaming()

                async def complete(self, system, payload, *, timeout, on_activity=None):
                    if payload.get("phase", "").startswith("handoff_"):
                        return await history_calls(config.role, system, payload, "offline")
                    return await normal.complete(system, payload, timeout=timeout, on_activity=on_activity)
                async def close(self):
                    await normal.close()
            return Model()

        def optimizer_factory(configs, options):
            engine = Optimizer(configs, options, factory=factory)
            self.engines.append(engine)
            return engine
        return OptimizerApp(root=self.root, config_loader=lambda root: self.configs,
                            optimizer_factory=optimizer_factory)

    async def wait_for(self, pilot, condition):
        for _ in range(150):
            await pilot.pause(0.01)
            if condition():
                return
        screen = pilot.app.screen
        detail = screen.query_one("#window-error").render() if isinstance(screen, WindowScreen) else type(screen).__name__
        self.fail(f"integration condition did not arrive: {detail}")

    async def test_parallel_activity_cancel_preserves_success_and_failure_report(self):
        a_started, b_started = asyncio.Event(), asyncio.Event()
        callbacks = []
        class MonitoredScripts(DialogueScripts):
            def factory(self, config):
                base = super().factory(config)
                class Model:
                    def disable_streaming(self):
                        base.disable_streaming()
                    async def complete(self, system, payload, *, timeout, on_activity=None):
                        on_activity(ModelActivity("mode", mode="streaming"))
                        if payload.get("phase") == "dialogue_clarification":
                            on_activity(ModelActivity("output", 5))
                            on_activity(ModelActivity("validating", finish_reason="stop"))
                            return await base.complete(system, payload, timeout=timeout, on_activity=on_activity)
                        callbacks.append(on_activity)
                        on_activity(ModelActivity("reasoning" if config.role == "a" else "output", 8))
                        (a_started if config.role == "a" else b_started).set()
                        await asyncio.Event().wait()
                    async def close(self):
                        await base.close()
                return Model()
        scripts = MonitoredScripts()
        previous = "上次成功提示词"
        (self.root / "last_optimized_prompt.md").write_text(previous, encoding="utf-8")
        app = self.app(scripts)
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#message-input", TextArea).load_text(ORIGINAL)
            await pilot.press("f2")
            await self.wait_for(pilot, lambda: a_started.is_set() and b_started.is_set())
            status = str(app.query_one("#status", Static).content)
            self.assertIn("A 正在思考", status)
            self.assertIn("B 正在接收结果", status)
            self.assertEqual(app.query_one("#prompt-output", TextArea).text, "")
            app.action_cancel_run()
            await self.wait_for(pilot, lambda: not app.busy)
            await app.workers.wait_for_complete()
            self.assertEqual(app.result.status, "cancelled")
            self.assertEqual((self.root / "last_optimized_prompt.md").read_text(encoding="utf-8"), previous)
            engine = self.engines[0]
            self.assertFalse(engine.activity_snapshot()["active_calls"])
            self.assertEqual(engine.calls[0]["finish_reason"], "stop")
            self.assertEqual([row["status"] for row in engine.calls], ["ok", "cancelled", "cancelled"])
            self.assertEqual([row["reasoning_chars"] for row in engine.calls], [0, 8, 0])
            self.assertEqual([row["output_chars"] for row in engine.calls], [5, 0, 8])
            before = str(app.query_one("#status", Static).content)
            for callback in callbacks:
                callback(ModelActivity("output", 999))
            self.assertEqual(str(app.query_one("#status", Static).content), before)
            report = json.loads(app.report_paths[-1].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "cancelled")
            self.assertFalse(report["metadata"]["active_calls"])
            self.assertCountEqual(scripts.closed, ["a", "b"])

    async def test_real_four_round_dialogue_pause_resume_and_report_checkpoints(self):
        scripts = DialogueScripts(clarification=[ask(question(text=f"第 {n} 个关键问题？")) for n in range(1, 5)])
        app = self.app(scripts)
        async with app.run_test(size=(120, 46)) as pilot:
            app.query_one("#message-input", TextArea).load_text(ORIGINAL)
            await pilot.press("f2")
            for n in range(1, 4):
                await self.wait_for(pilot, lambda: isinstance(app.screen, QuestionScreen)
                                    and app.screen.batch.round_number == n)
                app.screen.query_one("#answer-text-0", TextArea).load_text(f"第 {n} 个明确回答")
                app.screen.action_answer()
            await self.wait_for(pilot, lambda: isinstance(app.screen, QuestionScreen)
                                and app.screen.batch.round_number == 4)
            pending_id = app.screen.batch.questions[0].id
            self.assertEqual(len(self.engines[0].calls), 4)
            self.assertFalse(app.query("#continue-rounds"))
            app.screen.action_pause()
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(app.result.status, "needs_clarification")
            self.assertEqual(app.controller.snapshot()["status"], "paused")
            self.assertTrue(app.paused)
            self.assertEqual(str(app.query_one("#send").label), "继续对话 F2")
            self.assertEqual(len(self.engines[0].calls), 4)
            self.assertFalse((self.root / "last_optimized_prompt.md").exists())
            await pilot.press("f2")
            await self.wait_for(pilot, lambda: isinstance(app.screen, QuestionScreen))
            self.assertEqual(len(self.engines[0].calls), 4, "continue must show existing questions without a model request")
            self.assertEqual(app.screen.batch.questions[0].id, pending_id)
            app.screen.query_one("#answer-text-0", TextArea).load_text("最后的验收要求")
            app.screen.action_answer()
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(app.result.status, "ready")
            self.assertEqual(len(self.engines), 1)
            self.assertEqual(len(self.engines[0].calls), 8)
            self.assertEqual(app.controller.snapshot()["total_rounds"], 4)
            self.assertEqual(app.controller.snapshot()["original_request"], ORIGINAL)
            self.assertTrue((self.root / "last_optimized_prompt.md").exists())
            data = [json.loads(path.read_text(encoding="utf-8")) for path in app.report_paths]
            self.assertTrue(any(r["metadata"]["dialogue"]["status"] == "waiting" for r in data))
            self.assertTrue(any(r["metadata"]["dialogue"]["status"] == "paused" for r in data))
            self.assertTrue(any(r["status"] == "ready" for r in data))

    async def test_paused_supplement_is_rechecked_and_generates_without_reopening_old_question(self):
        scripts = DialogueScripts(clarification=[ask(question()), sufficient()])
        app = self.app(scripts)
        async with app.run_test(size=(120, 42)) as pilot:
            app.query_one("#message-input", TextArea).load_text(ORIGINAL)
            app.action_send()
            await self.wait_for(pilot, lambda: isinstance(app.screen, QuestionScreen))
            app.screen.action_pause()
            await self.wait_for(pilot, lambda: not app.busy)
            before = app.controller.snapshot()
            app.query_one("#message-input", TextArea).load_text("只检查异常处理，输出问题列表。")
            await pilot.press("f2")
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(app.result.status, "ready")
            self.assertEqual(len(self.engines), 1)
            self.assertEqual(app.result.metadata["request_count"], 5)
            self.assertIn("只检查异常处理", scripts.calls[1][2]["latest_updates"][0]["text"])
            self.assertEqual(app.controller.snapshot()["session_id"], before["session_id"])
            self.assertEqual(len(app.controller.snapshot()["updates"]), 1)
            self.assertEqual(app.controller.snapshot()["total_rounds"], 0)
            self.assertEqual(str(app.query_one("#send").label), "发送 F2")

    async def test_paused_empty_send_rechecks_changed_material(self):
        material = self.root / "material.md"
        material.write_text("新材料：验收为中文问题列表。", encoding="utf-8")
        scripts = DialogueScripts(clarification=[ask(question(text="旧材料的问题？")), sufficient()])
        app = self.app(scripts)
        async with app.run_test(size=(120, 42)) as pilot:
            app.query_one("#message-input", TextArea).load_text(ORIGINAL)
            app.action_send()
            await self.wait_for(pilot, lambda: isinstance(app.screen, QuestionScreen))
            app.screen.action_pause()
            await self.wait_for(pilot, lambda: not app.busy)
            before = app.controller.snapshot()
            app.query_one("#material-path", Input).value = str(material)
            app.action_add_material()
            await pilot.pause()
            self.assertEqual(len(app.materials), 1)
            self.assertEqual(app.query_one("#message-input", TextArea).text, "")
            await pilot.press("f2")
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(app.result.status, "ready")
            self.assertEqual(len(self.engines), 1)
            self.assertEqual(app.result.metadata["request_count"], 5)
            self.assertIn("新材料", str(scripts.calls[1][2]["reference_files"]))
            after = app.controller.snapshot()
            self.assertEqual(after["confirmed_request"], ORIGINAL)
            self.assertEqual(after["updates"], [])
            self.assertGreater(after["revision"], before["revision"])
            self.assertNotEqual(after["fingerprint"], before["fingerprint"])
            self.assertTrue((self.root / "last_optimized_prompt.md").exists())

    async def test_paused_empty_send_rebuilds_changed_history_before_clarification(self):
        windows = {"version": 1, "roles": {role: {
            **model_identity(self.configs[role]), "context_window": 128000} for role in required_roles()}}
        (self.root / "context_windows.json").write_text(json.dumps(windows), encoding="utf-8")
        history_calls = ScriptedCalls(current=True)
        scripts = DialogueScripts(clarification=[ask(question(text="旧历史的问题？")), sufficient()])
        app = self.app(scripts, history_calls=history_calls)
        async with app.run_test(size=(120, 42)) as pilot:
            app.query_one("#history-input", TextArea).load_text("用户：旧历史甲，只读，不改文件。")
            app.query_one("#message-input", TextArea).load_text(ORIGINAL)
            app.action_send()
            await self.wait_for(pilot, lambda: isinstance(app.screen, QuestionScreen))
            app.screen.action_pause()
            await self.wait_for(pilot, lambda: not app.busy)
            before = app.controller.snapshot()
            app.query_one("#history-input", TextArea).load_text("用户：新历史乙，只读，不改文件。")
            await pilot.pause()
            await pilot.press("f2")
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(app.result.status, "ready")
            self.assertEqual(len(self.engines), 1)
            self.assertEqual(len(history_calls.calls), 4)
            self.assertIn("新历史乙", str(history_calls.calls[2][2]["stage_input"]))
            self.assertIn("新历史乙", str(scripts.calls[1][2]["handoff_context"]))
            self.assertEqual(scripts.calls[1][2]["pending_questions"], [])
            self.assertEqual(app.result.metadata["request_count"], 9)
            after = app.controller.snapshot()
            self.assertEqual(after["session_id"], before["session_id"])
            self.assertEqual(after["confirmed_request"], ORIGINAL)
            self.assertEqual(after["updates"], [])
            self.assertGreater(after["revision"], before["revision"])
            self.assertNotEqual(after["fingerprint"], before["fingerprint"])

    async def test_editing_during_real_generation_blocks_old_result_publication(self):
        gate = asyncio.Event()
        entered = asyncio.Event()
        async def slow_generation(payload):
            entered.set()
            await gate.wait()
            return draft(payload["original_request"])
        scripts = DialogueScripts(a=[slow_generation])
        app = self.app(scripts)
        last = self.root / "last_optimized_prompt.md"
        last.write_text("上次成功", encoding="utf-8")
        async with app.run_test(size=(120, 38)) as pilot:
            app.query_one("#message-input", TextArea).load_text(ORIGINAL)
            app.action_send()
            await self.wait_for(pilot, entered.is_set)
            app.query_one("#message-input", TextArea).load_text("补充：输出问题清单")
            gate.set()
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(last.read_text(encoding="utf-8"), "上次成功")
            self.assertTrue(app.query_one("#copy-prompt").disabled)
            self.assertTrue(app.report_paths)
            app.action_send()
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(len(self.engines), 1)
            self.assertEqual(app.result.metadata["request_count"], 8)
            self.assertIn(ORIGINAL, last.read_text(encoding="utf-8"))
            self.assertIn("输出问题清单", last.read_text(encoding="utf-8"))

    async def test_real_cancel_closes_session_and_saves_report(self):
        entered = asyncio.Event()
        async def waiting(payload):
            entered.set()
            await asyncio.Event().wait()
        app = self.app(DialogueScripts(a=[waiting]))
        last = self.root / "last_optimized_prompt.md"
        last.write_text("上次成功", encoding="utf-8")
        async with app.run_test(size=(120, 38)) as pilot:
            app.query_one("#message-input", TextArea).load_text(ORIGINAL)
            app.action_send()
            await self.wait_for(pilot, entered.is_set)
            app.action_cancel_run()
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(app.result.status, "cancelled")
            self.assertTrue(app.controller.snapshot()["closed"])
            self.assertEqual(last.read_text(encoding="utf-8"), "上次成功")
            self.assertTrue(any(json.loads(path.read_text(encoding="utf-8"))["status"] == "cancelled"
                                for path in app.report_paths))

    async def test_missing_window_modal_configures_then_uses_real_handoff_engine(self):
        history_calls = ScriptedCalls(current=True)
        app = self.app(DialogueScripts(), history_calls=history_calls)
        async with app.run_test(size=(120, 42)) as pilot:
            app.query_one("#history-input", TextArea).load_text("用户：只读，不改文件。")
            app.query_one("#message-input", TextArea).load_text(ORIGINAL)
            app.action_send()
            await self.wait_for(pilot, lambda: isinstance(app.screen, WindowScreen))
            await pilot.pause()
            self.assertEqual(len(history_calls.calls), 0)
            for role in required_roles():
                app.screen.query_one(f"#window-{role}", Input).value = "128000"
            self.assertTrue(await pilot.click("#windows-save", offset=(5, 1)))
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(app.result.status, "ready")
            self.assertNotIn("max_requests", app.result.metadata["limits"])
            self.assertNotIn("total_timeout", app.result.metadata["limits"])
            self.assertEqual(app.result.metadata["request_count"], 6)
            self.assertTrue(app.result.metadata["handoff"]["coverage_complete"])
            public = json.loads((self.root / "context_windows.json").read_text(encoding="utf-8"))
            self.assertEqual(set(public["roles"]), set(required_roles()))
            self.assertTrue(all(public["roles"][role]["context_window"] == 128000 for role in required_roles()))
            self.assertNotIn("api_key", str(public))

    async def test_history_drives_normal_handoff_normal_without_resetting_session_or_budget(self):
        windows = {"version": 1, "roles": {role: {
            **model_identity(self.configs[role]), "context_window": 128000} for role in required_roles()}}
        (self.root / "context_windows.json").write_text(json.dumps(windows), encoding="utf-8")
        history_calls = ScriptedCalls(current=True)
        scripts = DialogueScripts()
        app = self.app(scripts, history_calls=history_calls)
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#token-budget", Input).value = "5000"
            app.query_one("#message-input", TextArea).load_text(ORIGINAL)
            await pilot.press("f2")
            await self.wait_for(pilot, lambda: not app.busy)
            before = app.controller.snapshot()
            self.assertEqual(app.result.metadata["workflow"], "normal")
            self.assertEqual(app.result.metadata["request_count"], 4)
            self.assertNotIn("max_requests", app.result.metadata["limits"])
            self.assertNotIn("total_timeout", app.result.metadata["limits"])
            previous_tokens = app.result.metadata["known_total_tokens"]
            for history, workflow, expected_requests in (("用户：保持只读。", "handoff", 10),
                                                         (" \n ", "normal", 14),
                                                         ("用户：重新加入历史，保持只读。", "handoff", 20)):
                app.query_one("#history-input", TextArea).load_text(history)
                await pilot.pause()
                self.assertEqual(app.query_one("#message-input", TextArea).text, "")
                await pilot.press("f2")
                await self.wait_for(pilot, lambda: not app.busy)
                self.assertEqual(app.result.status, "ready")
                metadata = app.result.metadata
                self.assertEqual(metadata["workflow"], workflow)
                self.assertEqual(metadata["request_count"], expected_requests)
                self.assertNotIn("max_requests", metadata["limits"])
                self.assertNotIn("total_timeout", metadata["limits"])
                self.assertEqual(metadata["limits"]["token_budget"], 5000)
                self.assertGreater(metadata["known_total_tokens"], previous_tokens)
                previous_tokens = metadata["known_total_tokens"]
                after = app.controller.snapshot()
                self.assertEqual(after["session_id"], before["session_id"])
                self.assertEqual(after["confirmed_request"], ORIGINAL)
                self.assertEqual(after["updates"], [])
                self.assertGreater(after["revision"], before["revision"])
                before = after
                if workflow == "normal":
                    self.assertNotIn("handoff", metadata)
                    self.assertIsNone(app.controller.optimizer.context_windows)
                    self.assertIsNone(app.controller.optimizer.handoff_context)
                    self.assertEqual(after["pending_questions"], [])
            self.assertEqual(len(self.engines), 1)
            self.assertEqual(len(history_calls.calls), 4)

    async def test_invalid_new_history_and_cancelled_window_keep_existing_session_and_success(self):
        app = self.app(DialogueScripts(), history_calls=ScriptedCalls(current=True))
        async with app.run_test(size=(120, 42)) as pilot:
            app.query_one("#message-input", TextArea).load_text(ORIGINAL)
            await pilot.press("f2")
            await self.wait_for(pilot, lambda: not app.busy)
            controller = app.controller
            before = controller.snapshot()
            last = self.root / "last_optimized_prompt.md"
            previous_prompt = last.read_text(encoding="utf-8")
            app.query_one("#history-source", Select).value = "files"
            app.query_one("#history-input", TextArea).load_text(str(self.root / "missing.md"))
            await pilot.pause()
            await pilot.press("f2")
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(app.result.status, "failed")
            self.assertIs(app.controller, controller)
            self.assertEqual(controller.snapshot(), before)
            self.assertEqual(controller.optimizer.metadata()["request_count"], 4)
            self.assertEqual(last.read_text(encoding="utf-8"), previous_prompt)
            app.query_one("#history-source", Select).value = "paste"
            app.query_one("#history-input", TextArea).load_text("用户：只读检查。")
            await pilot.pause()
            await pilot.press("f2")
            await self.wait_for(pilot, lambda: isinstance(app.screen, WindowScreen))
            await pilot.press("escape")
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(app.result.status, "cancelled")
            self.assertIs(app.controller, controller)
            self.assertEqual(controller.snapshot(), before)
            self.assertFalse(controller.snapshot()["closed"])
            self.assertEqual(controller.optimizer.metadata()["request_count"], 4)
            self.assertEqual(last.read_text(encoding="utf-8"), previous_prompt)
            await pilot.press("f2")
            await self.wait_for(pilot, lambda: isinstance(app.screen, WindowScreen))
            app.action_cancel_run()
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(app.result.status, "cancelled")
            self.assertIs(app.controller, controller)
            self.assertEqual(controller.snapshot(), before)
            self.assertFalse(controller.snapshot()["closed"])
            self.assertEqual(controller.optimizer.metadata()["request_count"], 4)
            self.assertEqual(last.read_text(encoding="utf-8"), previous_prompt)
            await pilot.press("f2")
            await self.wait_for(pilot, lambda: isinstance(app.screen, WindowScreen))
            for role in required_roles():
                app.screen.query_one(f"#window-{role}", Input).value = "128000"
            self.assertTrue(await pilot.click("#windows-save", offset=(5, 1)))
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(app.result.status, "ready")
            self.assertIs(app.controller, controller)
            self.assertEqual(controller.snapshot()["updates"], [])
            self.assertEqual(app.result.metadata["request_count"], 10)

    async def test_clearing_history_after_pause_discards_old_questions_and_window_binding(self):
        windows = {"version": 1, "roles": {role: {
            **model_identity(self.configs[role]), "context_window": 128000} for role in required_roles()}}
        (self.root / "context_windows.json").write_text(json.dumps(windows), encoding="utf-8")
        scripts = DialogueScripts(clarification=[ask(question(text="旧历史的待确认项？")), sufficient()])
        app = self.app(scripts, history_calls=ScriptedCalls(current=True))
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#history-input", TextArea).load_text("用户：保持只读。")
            app.query_one("#message-input", TextArea).load_text(ORIGINAL)
            await pilot.press("f2")
            await self.wait_for(pilot, lambda: isinstance(app.screen, QuestionScreen))
            app.screen.action_pause()
            await self.wait_for(pilot, lambda: not app.busy)
            app.query_one("#history-input", TextArea).load_text("")
            await pilot.pause()
            await pilot.press("f2")
            await self.wait_for(pilot, lambda: not app.busy)
            self.assertEqual(app.result.status, "ready")
            self.assertEqual(scripts.calls[1][2]["pending_questions"], [])
            self.assertIsNone(scripts.calls[1][2].get("handoff_context"))
            self.assertEqual(app.result.metadata["workflow"], "normal")
            self.assertNotIn("max_requests", app.result.metadata["limits"])
            self.assertIsNone(app.controller.optimizer.context_windows)
            self.assertEqual(app.controller.snapshot()["updates"], [])


if __name__ == "__main__":
    unittest.main()
