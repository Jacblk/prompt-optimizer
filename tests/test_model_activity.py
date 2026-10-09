"""Activity accounting and wait policy, with controlled clocks and offline models."""
import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

from optimizer_config import ConfigurationError, ModelConfig, model_config
from optimizer_dialogue import DialogueSession
from optimizer_engine import BudgetExceeded, Optimizer, RunOptions
from optimizer_models import ModelActivity, ModelCallError, ModelReply, StreamingUnsupportedError, TransientModelError


class ActivityTests(unittest.IsolatedAsyncioTestCase):
    def make_runner(self, actions, *, retries=0, token_budget=None):
        models = {}
        events = []
        configs = {role: ModelConfig(role, "offline", "http://127.0.0.1:9/v1", "placeholder",
                                     timeout=0.01, slow_warning_seconds=90) for role in ("a", "b", "judge")}
        def factory(config):
            class Model:
                streaming = True
                disabled = 0
                closed = False
                callbacks = []
                async def complete(self, system, payload, *, timeout, on_activity=None):
                    self.callbacks = [*self.callbacks, on_activity]
                    on_activity(ModelActivity("mode", mode="streaming" if self.streaming else "non_streaming"))
                    action = actions[config.role].pop(0)
                    if isinstance(action, Exception):
                        raise action
                    return await action(on_activity) if callable(action) else action
                def disable_streaming(self):
                    self.streaming = False
                    self.disabled += 1
                async def close(self):
                    self.closed = True
            model = Model()
            models[config.role] = model
            return model
        runner = Optimizer(configs, RunOptions(retries=retries, token_budget=token_budget), factory=factory)
        runner.started = time.monotonic()
        runner.dialogue_session = DialogueSession(session_id="offline-session", revision=1)
        runner.dialogue_event = events.append
        self.addAsyncCleanup(runner.aclose)
        return runner, models, events

    async def test_long_silence_warns_without_terminating_then_activity_clears_warning(self):
        entered, release = asyncio.Event(), asyncio.Event()
        callback = []
        async def waiting(on_activity):
            callback.append(on_activity)
            entered.set()
            await release.wait()
            return ModelReply("完整结果", 5, 4, 9)
        runner, models, events = self.make_runner({"judge": [waiting]})
        clock = [0.0]
        runner._activity_clock = lambda: clock[0]
        task = asyncio.create_task(runner._call("judge", "system", {}, "review"))
        try:
            await entered.wait()
            await asyncio.sleep(0.03)  # Longer than the obsolete whole-request timeout.
            clock[0] = 120
            status = runner.activity_snapshot()["active_calls"][0]
            self.assertTrue(status["slow_waiting"])
            self.assertEqual(status["activity_state"], "waiting")
            self.assertFalse(task.done())
            callback[0](ModelActivity("reasoning", 8))
            status = runner.activity_snapshot()["active_calls"][0]
            self.assertFalse(status["slow_waiting"])
            self.assertEqual(status["activity_state"], "thinking")
            self.assertEqual(status["first_activity_seconds"], 120)
            clock[0] = 220
            self.assertTrue(runner.activity_snapshot()["active_calls"][0]["slow_waiting"])
            callback[0](ModelActivity("output", 4))
            self.assertFalse(runner.activity_snapshot()["active_calls"][0]["slow_waiting"])
            callback[0](ModelActivity("validating", finish_reason="stop"))
            self.assertEqual(runner.activity_snapshot()["active_calls"][0]["activity_state"], "validating")
            release.set()
            self.assertEqual((await task).text, "完整结果")
            self.assertEqual(runner.calls[0]["reasoning_chars"], 8)
            self.assertEqual(runner.calls[0]["output_chars"], 4)
            self.assertEqual(runner.calls[0]["elapsed_seconds"], 220)
            self.assertEqual(runner.calls[0]["finish_reason"], "stop")
            self.assertEqual(runner.known_tokens, 9)
            self.assertFalse(runner.activity_snapshot()["active_calls"])
            self.assertTrue(any(event["kind"] == "activity" for event in events))
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_request_duration_and_activity_offsets_share_the_same_start(self):
        clock = [0]
        async def response(callback):
            clock[0] = 3
            callback(ModelActivity("output", 4))
            clock[0] = 5
            return ModelReply("完整结果", 2, 2, 4)
        runner, _, events = self.make_runner({"a": [response]})
        runner._activity_clock = lambda: clock[0]
        def display(event):
            events.append(event)
            if event["kind"] == "stage":
                clock[0] = 2  # Display/observer cost after the request was reserved.
        runner.dialogue_event = display
        await runner._call("a", "system", {}, "generate")
        self.assertEqual(runner.calls[0]["elapsed_seconds"], 5)
        self.assertEqual(runner.calls[0]["first_activity_seconds"], 3)
        self.assertEqual(runner.calls[0]["last_activity_seconds"], 3)

    async def test_parallel_a_b_progress_and_completion_are_independent(self):
        a_entered, b_entered, a_release, b_release = [asyncio.Event() for _ in range(4)]
        async def a(callback):
            callback(ModelActivity("reasoning", 10))
            a_entered.set()
            await a_release.wait()
            return ModelReply("甲", 1, 1, 2)
        async def b(callback):
            callback(ModelActivity("output", 6))
            b_entered.set()
            await b_release.wait()
            return ModelReply("乙", 1, 1, 2)
        runner, _, _ = self.make_runner({"a": [a], "b": [b]})
        tasks = [asyncio.create_task(runner._call(role, "system", {}, "generate")) for role in ("a", "b")]
        try:
            await asyncio.gather(a_entered.wait(), b_entered.wait())
            active = {row["role"]: row for row in runner.activity_snapshot()["active_calls"]}
            self.assertEqual((active["a"]["activity_state"], active["b"]["activity_state"]), ("thinking", "receiving"))
            self.assertNotEqual(active["a"]["request_id"], active["b"]["request_id"])
            b_release.set()
            await tasks[1]
            self.assertEqual([row["role"] for row in runner.activity_snapshot()["active_calls"]], ["a"])
            a_release.set()
            await tasks[0]
            self.assertEqual(len(runner.calls), 2)
        finally:
            a_release.set()
            b_release.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_retry_ignores_old_callback_and_does_not_merge_partial_counts(self):
        callbacks = []
        async def failed(callback):
            callbacks.append(callback)
            callback(ModelActivity("output", 100))
            raise TransientModelError("offline disconnect")
        async def succeeded(callback):
            callbacks[0](ModelActivity("output", 500))
            callback(ModelActivity("output", 4))
            return ModelReply("新的结果", 2, 2, 4)
        runner, _, events = self.make_runner({"a": [failed, succeeded]}, retries=1)
        reply = await runner._call("a", "system", {}, "generate")
        self.assertEqual(reply.text, "新的结果")
        self.assertEqual([row["output_chars"] for row in runner.calls], [100, 4])
        self.assertEqual([row["attempt"] for row in runner.calls], [1, 2])
        self.assertEqual(runner.unknown_usage, 1)
        self.assertTrue(any(any(row["activity_state"] == "retrying" for row in event.get("active_calls", []))
                            for event in events))

    async def test_explicit_stream_rejection_has_one_counted_fallback_and_is_cached(self):
        runner, models, _ = self.make_runner({"a": [StreamingUnsupportedError("offline unsupported stream"),
                                                   ModelReply("普通结果", 2, 2, 4), ModelReply("下一结果", 2, 2, 4)]})
        self.assertEqual((await runner._call("a", "system", {}, "generate")).text, "普通结果")
        self.assertEqual([row["status"] for row in runner.calls], ["stream_unsupported", "ok"])
        self.assertEqual(runner.calls[1]["call_mode"], "non_streaming")
        await runner._call("a", "system", {}, "repair")
        self.assertEqual(len(runner.calls), 3)
        self.assertEqual(models["a"].disabled, 1)
        self.assertEqual(runner.unknown_usage, 1)
        self.assertEqual(runner.known_tokens, 8)

    async def test_more_than_old_retry_cap_and_stream_fallback_have_separate_allowances(self):
        actions = [StreamingUnsupportedError("offline rejection")]
        actions += [TransientModelError("offline network failure") for _ in range(7)]
        actions += [ModelReply("complete", 1, 1, 2)]
        runner, models, _ = self.make_runner({"a": actions}, retries=7)
        with patch("optimizer_engine.asyncio.sleep", new_callable=AsyncMock):
            reply = await runner._call("a", "system", {}, "generate")
        self.assertEqual(reply.text, "complete")
        self.assertEqual(len(runner.calls), 9)
        self.assertEqual([row["attempt"] for row in runner.calls], list(range(1, 10)))
        self.assertEqual(sum(row["status"] == "stream_unsupported" for row in runner.calls), 1)
        self.assertEqual(sum(row["status"] == "transient_error" for row in runner.calls), 7)
        self.assertEqual(models["a"].disabled, 1)
        exhausted, _, _ = self.make_runner({"a": [TransientModelError("offline") for _ in range(9)]}, retries=7)
        with patch("optimizer_engine.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaises(ModelCallError):
                await exhausted._call("a", "system", {}, "generate")
        self.assertEqual(len(exhausted.calls), 8)

    async def test_large_attempt_count_does_not_overflow_retry_backoff(self):
        actions = [TransientModelError("offline failure") for _ in range(1100)]
        actions += [ModelReply("complete", 1, 1, 2)]
        runner, _, _ = self.make_runner({"a": actions}, retries=1100)
        with patch("optimizer_engine.asyncio.sleep", new_callable=AsyncMock) as sleep:
            self.assertEqual((await runner._call("a", "system", {}, "generate")).text, "complete")
        self.assertEqual(len(runner.calls), 1101)
        self.assertTrue(all(0 <= args.args[0] <= 4 for args in sleep.await_args_list))

    async def test_fallback_is_blocked_by_unknown_usage_token_budget(self):
        runner, models, _ = self.make_runner({"a": [StreamingUnsupportedError("offline unsupported stream"),
                                                   ModelReply("不能调度", 2, 2, 4)]}, token_budget=100)
        with self.assertRaises(BudgetExceeded):
            await runner._call("a", "system", {}, "generate")
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(models["a"].disabled, 0)
        self.assertFalse(runner.activity_snapshot()["active_calls"])

    async def test_network_retries_remain_bounded_and_missing_usage_is_unknown(self):
        runner, _, _ = self.make_runner({"a": [TransientModelError("offline network"),
                                              TransientModelError("offline network")]}, retries=1)
        with self.assertRaises(ModelCallError):
            await runner._call("a", "system", {}, "generate")
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(runner.unknown_usage, 2)
        runner, _, _ = self.make_runner({"a": [ModelReply("有效结果")]})
        await runner._call("a", "system", {}, "generate")
        self.assertEqual(runner.known_tokens, 0)
        self.assertEqual(runner.unknown_usage, 1)
        self.assertIsNone(runner.metadata()["total_tokens"])

    async def test_cancel_clears_monitor_and_late_callbacks_are_ignored(self):
        entered = asyncio.Event()
        callback = []
        async def forever(on_activity):
            callback.append(on_activity)
            entered.set()
            await asyncio.Event().wait()
        runner, models, _ = self.make_runner({"a": [forever]})
        task = asyncio.create_task(runner._call("a", "system", {}, "generate"))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        callback[0](ModelActivity("reasoning", 100))
        self.assertEqual(runner.calls[0]["status"], "cancelled")
        self.assertEqual(runner.calls[0]["reasoning_chars"], 0)
        self.assertFalse(runner.activity_snapshot()["active_calls"])
        await runner.aclose()
        self.assertTrue(models["a"].closed)

    async def test_changed_revision_filters_activity_and_stale_monitor_events(self):
        entered, release = asyncio.Event(), asyncio.Event()
        callback = []
        async def waiting(on_activity):
            callback.append(on_activity)
            entered.set()
            await release.wait()
            return ModelReply("旧结果", 1, 1, 2)
        runner, _, events = self.make_runner({"a": [waiting]})
        task = asyncio.create_task(runner._call("a", "system", {}, "generate"))
        try:
            await entered.wait()
            runner.dialogue_session.revision = 2
            before = len(events)
            callback[0](ModelActivity("reasoning", 10))
            runner._emit_activity(runner.calls[0])
            self.assertEqual(len(events), before)
            self.assertFalse(runner.activity_snapshot()["active_calls"])
            release.set()
            await task
            self.assertEqual(runner.calls[0]["reasoning_chars"], 0)
            self.assertEqual(runner.calls[0]["revision"], 1)
            self.assertEqual(len(events), before)
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_invalid_or_zero_length_signals_do_not_reset_inactivity(self):
        entered, release = asyncio.Event(), asyncio.Event()
        callbacks = []
        async def waiting(callback):
            callbacks.append(callback)
            entered.set()
            await release.wait()
            return ModelReply("完成", 1, 1, 2)
        runner, _, _ = self.make_runner({"a": [waiting]})
        clock = [0]
        runner._activity_clock = lambda: clock[0]
        task = asyncio.create_task(runner._call("a", "system", {}, "generate"))
        try:
            await entered.wait()
            clock[0] = 120
            for activity in (ModelActivity("reasoning", 0), ModelActivity("output", -1),
                             ModelActivity("output", True), {"text": "must-not-be-retained"}):
                callbacks[0](activity)
            self.assertTrue(runner.activity_snapshot()["active_calls"][0]["slow_waiting"])
            self.assertIsNone(runner.calls[0]["first_activity_seconds"])
            self.assertNotIn("must-not-be-retained", json.dumps(runner.metadata()))
        finally:
            release.set()
            await task


class ActivityConfigTests(unittest.TestCase):
    def test_warning_configuration_and_network_timeout_are_independent(self):
        values = {"JUDGE_NAME": "offline", "JUDGE_API_KEY": "placeholder", "JUDGE_BASE_URL": "http://127.0.0.1:9/v1"}
        config = model_config(values, "JUDGE", "judge")
        self.assertEqual((config.timeout, config.slow_warning_seconds), (90, 90))
        config = model_config({**values, "JUDGE_TIMEOUT": "30", "JUDGE_SLOW_WARNING_SECONDS": "4500"}, "JUDGE", "judge")
        self.assertEqual((config.timeout, config.slow_warning_seconds), (30, 4500))
        for value in ("0", "-1", "nan", "inf", "private-invalid-value"):
            with self.subTest(value=value), self.assertRaises(ConfigurationError) as caught:
                model_config({**values, "JUDGE_SLOW_WARNING_SECONDS": value}, "JUDGE", "judge")
            self.assertNotIn(value, str(caught.exception))
