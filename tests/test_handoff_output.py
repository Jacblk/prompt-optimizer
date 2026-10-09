"""Regression for history reasoning exhausting the generation output allowance."""
import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

from optimizer_config import ConfigurationError, ModelConfig, handoff_model_config, model_config
from optimizer_engine import Optimizer, RunOptions
from optimizer_handoff import WindowExceeded, WindowLimits, prepare_history
from optimizer_models import ModelReply, OutputTruncatedError, StreamingUnsupportedError
from test_handoff import HistoryScripts, runner, windows
from test_optimizer import ORIGINAL, configs


def deepseek_configs():
    return {role: replace(config, name="deepseek-flash" if role == "a" else
                          "deepseek-v4-pro" if role == "judge" else config.name,
                          extra_body={"reasoning_effort": "high"})
            for role, config in configs().items()}


class HandoffSettingsTests(unittest.TestCase):
    def test_thinking_history_reserves_64k_and_preserves_explicit_high_effort(self):
        models = deepseek_configs()
        for role in ("a", "judge"):
            history = handoff_model_config(models[role])
            self.assertEqual(history.max_tokens, 65536)
            self.assertEqual(history.extra_body["reasoning_effort"], "high")
            self.assertEqual(models[role].max_tokens, 8192)
        self.assertIs(handoff_model_config(models["b"]), models["b"])
        ordinary = configs()["a"]
        self.assertIs(handoff_model_config(ordinary), ordinary)

    def test_nonthinking_and_larger_explicit_limits_keep_their_settings(self):
        model = deepseek_configs()["a"]
        for extra in ({"reasoning_effort": "none"}, {"thinking": {"type": "disabled"}}):
            self.assertEqual(handoff_model_config(replace(model, extra_body=extra)).max_tokens, 8192)
        self.assertEqual(handoff_model_config(replace(model, max_tokens=131072)).max_tokens, 131072)

    def test_role_overrides_are_validated_and_do_not_mutate_generation(self):
        values = {"GENERATOR_A_NAME": "deepseek-flash", "GENERATOR_A_API_KEY": "offline-placeholder",
                  "GENERATOR_A_BASE_URL": "https://offline.invalid/v1",
                  "GENERATOR_A_EXTRA_BODY": '{"reasoning_effort":"high","thinking":{"type":"enabled"}}',
                  "GENERATOR_A_HANDOFF_MAX_TOKENS": "32768", "GENERATOR_A_HANDOFF_REASONING_EFFORT": "low"}
        model = model_config(values, "GENERATOR_A", "a")
        history = handoff_model_config(model)
        self.assertEqual(history.max_tokens, 32768)
        self.assertEqual(history.extra_body["reasoning_effort"], "low")
        self.assertEqual(model.extra_body["reasoning_effort"], "high")
        history.extra_body["thinking"]["type"] = "disabled"
        self.assertEqual(model.extra_body["thinking"]["type"], "enabled")
        for key, value in (("HANDOFF_MAX_TOKENS", "0"), ("HANDOFF_MAX_TOKENS", "-1"),
                           ("HANDOFF_MAX_TOKENS", "nan"), ("HANDOFF_REASONING_EFFORT", "invalid-private-value")):
            with self.subTest(key=key, value=value), self.assertRaises(ConfigurationError) as caught:
                model_config(values | {"GENERATOR_A_" + key: value}, "GENERATOR_A", "a")
            self.assertNotIn("invalid-private-value", str(caught.exception))

    def test_insufficient_history_output_window_is_rejected_before_any_call(self):
        models = deepseek_configs()
        with self.assertRaises(ConfigurationError):
            WindowLimits.from_config(windows(models, a=64000), models)


class HandoffOutputFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_dialogue_truncation_is_not_reported_as_a_fidelity_rejection(self):
        error = OutputTruncatedError("a 输出上限 8192 token。",
                                     usage={"input_tokens": 10, "output_tokens": 8192, "total_tokens": 8202})
        scripts = HistoryScripts(handoff_extract=[error, error])
        engine = runner(scripts)
        try:
            result = await engine.run_dialogue(ORIGINAL)
        finally:
            await engine.aclose()
        self.assertEqual(result.status, "handoff_output_truncated", result.reason)
        self.assertEqual(result.metadata["handoff"]["status"], result.status)
        self.assertIn("8192", result.reason)
        self.assertIn("创建新会话", result.reason)
        self.assertIsNone(result.metadata["handoff"]["snapshot"])
        self.assertEqual(result.metadata["known_total_tokens"], 16404)
        self.assertEqual(len(scripts.calls), 2)
        self.assertTrue(all(role == "a" for role, *_ in scripts.calls))

    async def test_window_check_uses_history_reservation_at_actual_call_boundary(self):
        models = deepseek_configs()
        scripts = HistoryScripts()
        engine = Optimizer(models, RunOptions(), factory=scripts.factory)
        engine.windows = WindowLimits.from_config(windows(models), models)
        try:
            with self.assertRaises(WindowExceeded):
                await engine._call("a", "system", {"text": "x" * 55000}, "extract-1_summarize")
            self.assertEqual(scripts.calls, [])
            self.assertEqual(engine.calls, [])
        finally:
            await engine.aclose()

    async def test_streaming_fallback_carries_over_to_separate_generation_client(self):
        made = []
        def factory(config):
            class Fake:
                streaming = True
                closed = False
                def disable_streaming(self):
                    self.streaming = False
                async def complete(self, system, payload, *, timeout, on_activity=None):
                    if self.streaming:
                        raise StreamingUnsupportedError("offline stream rejection")
                    return ModelReply("{}", 1, 1, 2)
                async def close(self):
                    self.closed = True
            model = Fake()
            made.append((config, model))
            return model
        engine = Optimizer(deepseek_configs(), RunOptions(retries=0), factory=factory)
        try:
            await engine._call("a", "system", {}, "extract-1_summarize")
            await engine._call("a", "system", {}, "generate-a")
            self.assertEqual([config.max_tokens for config, _ in made], [65536, 8192])
            self.assertEqual(len(engine.calls), 3)
            self.assertEqual(engine.calls[-1]["call_mode"], "non_streaming")
        finally:
            await engine.aclose()
        self.assertTrue(all(model.closed for _, model in made))


class HandoffOutputWireTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        requests = self.requests
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                wire = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(wire)
                payload = json.loads(wire["messages"][1]["content"])
                role = {"deepseek-flash": "a", "deepseek-v4-pro": "judge", "model-b": "b"}[wire["model"]]
                limit = wire.get("max_tokens", wire.get("max_completion_tokens"))
                truncated = payload.get("phase") == "handoff_extract" and limit <= 8192
                content = "" if truncated else json.dumps(HistoryScripts.default(role, payload), ensure_ascii=False)
                response = {"id": "offline", "object": "chat.completion", "created": 0, "model": wire["model"],
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                                                                  "reasoning_content": "private-reasoning-must-not-be-saved"},
                                         "finish_reason": "length" if truncated else "stop"}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": limit if truncated else 4,
                                      "total_tokens": 10 + limit if truncated else 14}}
                body = json.dumps(response, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def engine(self, *, force_old_limit=False):
        models = {role: replace(config, base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
                                token_limit_field="max_completion_tokens" if role == "judge" else "max_tokens",
                                handoff_max_tokens=8192 if force_old_limit and role == "a" else None)
                  for role, config in deepseek_configs().items()}
        return Optimizer(models, RunOptions(), history=prepare_history(text="用户：只读，不改文件。"),
                         context_windows=windows(models))

    async def test_sdk_uses_64k_for_history_and_8k_for_generation_with_high_preserved(self):
        result = await self.engine().run(ORIGINAL)
        self.assertEqual(result.status, "ready", result.reason)
        self.assertEqual(len(self.requests), 5)
        for wire in self.requests:
            payload = json.loads(wire["messages"][1]["content"])
            history = payload.get("phase") in {"handoff_extract", "handoff_fidelity"}
            self.assertEqual(wire.get("max_tokens", wire.get("max_completion_tokens")), 65536 if history else 8192)
            self.assertEqual(wire["reasoning_effort"], "high")
        for call in result.metadata["calls"]:
            history = call["purpose"].startswith("extract-")
            self.assertEqual(call["max_tokens"], 65536 if history else 8192)
            self.assertEqual(call["window_check"]["reserved_output_tokens"], call["max_tokens"])
        self.assertTrue(result.metadata["handoff"]["coverage_complete"])

    async def test_old_output_ceiling_reproduces_failure_with_precise_status_and_usage(self):
        result = await self.engine(force_old_limit=True).run(ORIGINAL)
        self.assertEqual(result.status, "handoff_output_truncated", result.reason)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(result.metadata["known_total_tokens"], 16404)
        self.assertIsNone(result.optimized_prompt)
        self.assertIsNone(result.metadata["handoff"]["snapshot"])
        self.assertTrue(all(call["finish_reason"] == "length" for call in result.metadata["calls"]))
        self.assertNotIn("private-reasoning-must-not-be-saved", json.dumps(result.to_dict()))


if __name__ == "__main__":
    unittest.main()
