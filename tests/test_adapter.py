"""Exercise the installed SDK against a loopback server; no external API access."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from dataclasses import asdict
from unittest.mock import patch

from optimizer_config import ModelConfig, model_config
from optimizer_engine import BudgetExceeded, Optimizer, RunOptions
from optimizer_models import LangChainChatModel, OutputError, StreamingUnsupportedError, TransientModelError
from optimizer_prompts import generation_prompt, review_prompt
from optimizer_layers import analysis_prompt
from optimizer_review import evaluate_review


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.responses = []
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                test.requests.append((self.path, payload))
                status, response = test.responses.pop(0) if test.responses else (200, {
                    "id": "offline", "object": "chat.completion", "created": 0, "model": "returned-model",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "有效文本"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                })
                if callable(response):
                    response(self)
                    return
                if isinstance(response, list):
                    body = b"".join((f"data: {json.dumps(event, ensure_ascii=False)}\n\n" if isinstance(event, dict)
                                     else f"{event}\n\n").encode("utf-8") for event in response)
                    self.send_response(status)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = json.dumps(response).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    async def test_real_sdk_payload_and_usage(self):
        model = LangChainChatModel(ModelConfig("a", "test-model", self.url, "offline-placeholder",
                                               max_tokens=2048, json_mode=True,
                                               extra_body={"thinking": {"type": "disabled"}}))
        try:
            result = await model.complete('输出 JSON：{"literal":true}', {"original_request": "{payload}\nEND\n保留"}, timeout=3)
        finally:
            await model.close()
        path, payload = self.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertNotIn("max_completion_tokens", payload)
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["messages"][0]["content"], '输出 JSON：{"literal":true}')
        self.assertEqual(json.loads(payload["messages"][1]["content"])["original_request"], "{payload}\nEND\n保留")
        self.assertEqual((result.input_tokens, result.output_tokens, result.total_tokens), (10, 4, 14))
        self.assertEqual(result.returned_model, "returned-model")
        self.assertEqual(result.text, "有效文本")

    async def test_alternate_token_field(self):
        model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder",
                                               token_limit_field="max_completion_tokens", temperature=0.5))
        try:
            await model.complete("system", {}, timeout=3)
        finally:
            await model.close()
        payload = self.requests[0][1]
        self.assertEqual(payload["max_completion_tokens"], 8192)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["temperature"], 0.5)

    async def test_saved_output_above_old_cap_reaches_the_installed_sdk(self):
        from optimizer_config import load_models, read_environment, handoff_model_config
        from optimizer_settings import ConfigurationStore
        from test_configuration import seed_models
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_models(root)
            ConfigurationStore(root).apply({"models.a.base_url": self.url,
                "models.a.max_tokens": 384000, "models.a.handoff_max_tokens": 500000,
                "models.a.context_window": 1000000})
            config = load_models(read_environment(root / ".env"))["a"]
            for profile in (config, handoff_model_config(config)):
                model = LangChainChatModel(profile)
                try:
                    await model.complete("system", {}, timeout=3)
                finally:
                    await model.close()
        self.assertEqual([payload["max_tokens"] for _, payload in self.requests], [384000, 500000])

    async def test_loaded_reference_blocks_survive_the_actual_sdk_transport(self):
        from optimizer_documents import ReferenceFile, prepare_references
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "参考.py"
            text = 'def value():\n    return {"变量": "{payload}"}\n'
            path.write_text(text, encoding="utf-8")
            bundle = prepare_references([ReferenceFile(path)])
            material = {"original_request": "只参考结构，不修改文件。", "reference_files": bundle.payload()}
            model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder"))
            try:
                await model.complete(generation_prompt("a"), material, timeout=3)
            finally:
                await model.close()
        wire = self.requests[0][1]
        self.assertEqual(json.loads(wire["messages"][1]["content"]), material)
        self.assertIn(text, material["reference_files"][0]["block"])
        self.assertNotIn("tools", wire)

    async def test_official_review_reaches_real_sdk_once_without_wire_schema_changes(self):
        value = {"reviews": [{"candidate_id": "candidate_1", "verdict": "pass", "findings": [],
                               "clarity": 4, "conciseness": 4, "reason": "已检查。"}],
                 "action": "keep_original", "candidate_id": "candidate_1",
                 "clarification_questions": [], "reason": "保留。"}
        self.responses.append((200, {"id": "offline", "object": "chat.completion", "created": 0,
                                     "model": "offline-judge",
                                     "choices": [{"index": 0, "message": {"role": "assistant",
                                                  "content": json.dumps(value)}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}))
        model = LangChainChatModel(ModelConfig("judge", "offline-judge", self.url, "offline-placeholder",
                                               token_limit_field="max_completion_tokens", json_mode=True))
        replies = []
        async def request(system, payload):
            reply = await model.complete(system, payload, timeout=3)
            replies.append(reply)
            return reply
        material = {"original_request": '核对 {implementation}；只输出 {"ok":true}。',
                    "original_candidate_id": "candidate_1", "candidates": []}
        try:
            result = await evaluate_review(review_prompt(), material, request)
        finally:
            await model.close()
        self.assertEqual(result.model_dump(), value)
        self.assertEqual(len(self.requests), 1)
        payload = self.requests[0][1]
        self.assertEqual(payload["model"], "offline-judge")
        self.assertEqual(payload["max_completion_tokens"], 8192)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("tools", payload)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["messages"][0]["content"], review_prompt())
        self.assertEqual(json.loads(payload["messages"][1]["content"]), material)
        self.assertEqual((replies[0].input_tokens, replies[0].output_tokens, replies[0].total_tokens), (10, 4, 14))

    async def test_versioned_prompts_and_user_template_reach_sdk_without_interpolation(self):
        material = {'original_request': '按 {tone} 改写 {{body}}，输出 {"result": "文本"}。'}
        model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder"))
        try:
            systems = [generation_prompt(strategy) for strategy in ("a", "a", "b", "repair")]
            for system in systems + [review_prompt(), analysis_prompt()]:
                await model.complete(system, material, timeout=3)
                messages = self.requests[-1][1]["messages"]
                self.assertEqual(messages[0]["content"], system)
                self.assertEqual(json.loads(messages[1]["content"]), material)
        finally:
            await model.close()

    async def test_blank_temperature_omitted_for_each_role_and_model(self):
        for prefix, role, name in (("GENERATOR_A", "a", "deepseek-flash"),
                                   ("GENERATOR_B", "b", "step-5-preview"),
                                   ("JUDGE", "judge", "kimi-k3"),
                                   ("MODEL", "a", "kimi-k3")):
            with self.subTest(role=role, model=name):
                config = model_config({f"{prefix}_NAME": name, f"{prefix}_API_KEY": "offline-placeholder",
                                       f"{prefix}_BASE_URL": self.url, f"{prefix}_TEMPERATURE": "  "}, prefix, role)
                model = LangChainChatModel(config)
                try:
                    await model.complete("system", {}, timeout=3)
                finally:
                    await model.close()
                payload = self.requests[-1][1]
                self.assertEqual(payload["model"], name)
                self.assertNotIn("temperature", payload)

    async def test_explicit_zero_is_distinct_from_omitting_temperature(self):
        config = model_config({"MODEL_NAME": "adjustable-test-model", "MODEL_API_KEY": "offline-placeholder",
                               "MODEL_BASE_URL": self.url, "MODEL_TEMPERATURE": "0"}, "MODEL", "a")
        model = LangChainChatModel(config)
        try:
            await model.complete("system", {}, timeout=3)
        finally:
            await model.close()
        self.assertIn("temperature", self.requests[0][1])
        self.assertEqual(self.requests[0][1]["temperature"], 0)

    async def test_closing_one_model_does_not_close_another_at_same_endpoint(self):
        first = LangChainChatModel(ModelConfig("a", "model-a", self.url, "offline-placeholder"))
        second = LangChainChatModel(ModelConfig("b", "model-b", self.url, "offline-placeholder"))
        try:
            await asyncio.gather(first.complete("system", {}, timeout=3),
                                 second.complete("system", {}, timeout=3))
            await first.close()
            response = await second.complete("system", {}, timeout=3)
            self.assertEqual(response.total_tokens, 14)
            self.assertEqual(len(self.requests), 3)
        finally:
            await first.close()
            await second.close()

    async def test_truncation_is_not_success(self):
        self.responses.append((200, {"id": "offline", "object": "chat.completion", "created": 0, "model": "test",
                                     "choices": [{"index": 0, "message": {"role": "assistant", "content": "partial"},
                                                  "finish_reason": "length"}]}))
        model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder"))
        try:
            with self.assertRaises(OutputError):
                await model.complete("system", {}, timeout=3)
        finally:
            await model.close()

    async def test_truncated_response_retains_sanitized_usage_for_shared_budget(self):
        self.responses.append((200, {"id": "offline", "object": "chat.completion", "created": 0, "model": "test",
                                     "choices": [{"index": 0, "message": {"role": "assistant", "content": "do-not-log-partial-body"},
                                                  "finish_reason": "length"}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}))
        model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder"))
        try:
            with self.assertRaises(OutputError) as caught:
                await model.complete("system", {}, timeout=3)
        finally:
            await model.close()
        self.assertEqual(caught.exception.usage, {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})
        self.assertNotIn("do-not-log", str(caught.exception))

    async def test_sdk_retries_are_disabled_and_errors_are_sanitized(self):
        self.responses.append((429, {"error": {"message": "do-not-print-provider-details", "type": "rate_limit_error"}}))
        model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder"))
        try:
            with self.assertRaises(TransientModelError) as caught:
                await model.complete("system", {}, timeout=3)
        finally:
            await model.close()
        self.assertNotIn("do-not-print", str(caught.exception))
        self.assertEqual(len(self.requests), 1)

    @staticmethod
    def chunk(delta=None, *, finish=None, usage=None):
        return {"id": "offline-stream", "object": "chat.completion.chunk", "created": 0,
                "model": "returned-model", "choices": ([{"index": 0, "delta": delta or {},
                                                         "finish_reason": finish}] if delta is not None or finish else []),
                "usage": usage}

    async def test_reasoning_and_text_stream_preserve_usage_without_exposing_reasoning(self):
        secret_reasoning = "private-reasoning-must-never-be-saved"
        self.responses.append((200, [self.chunk({"role": "assistant", "content": ""}),
                                    self.chunk({"reasoning_content": secret_reasoning}),
                                    self.chunk({"content": '{"value":'}), self.chunk({"content": '"中文"}'}),
                                    self.chunk({}, finish="stop"),
                                    self.chunk(usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}),
                                    "data: [DONE]"]))
        events = []
        model = LangChainChatModel(ModelConfig("judge", "test", self.url, "offline-placeholder"))
        try:
            reply = await model.complete("system", {}, timeout=3, on_activity=events.append)
        finally:
            await model.close()
        self.assertEqual(json.loads(reply.text), {"value": "中文"})
        self.assertEqual(reply.total_tokens, 18)
        self.assertEqual(reply.returned_model, "returned-model")
        self.assertEqual([event.finish_reason for event in events if event.kind == "validating"], ["stop"])
        self.assertEqual(sum(event.characters for event in events if event.kind == "reasoning"), len(secret_reasoning))
        self.assertNotIn(secret_reasoning, json.dumps([asdict(event) for event in events]))
        self.assertNotIn(secret_reasoning, repr(reply))
        self.assertTrue(self.requests[0][1]["stream"])
        self.assertEqual(self.requests[0][1]["stream_options"], {"include_usage": True})

    async def test_reasoning_only_and_truncated_streams_do_not_return_partial_text(self):
        for delta, finish in (({"reasoning_content": "private-reasoning"}, "stop"),
                              ({"content": "private-partial-body"}, "length")):
            with self.subTest(finish=finish):
                self.responses.append((200, [self.chunk(delta), self.chunk({}, finish=finish),
                                            self.chunk(usage={"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}),
                                            "data: [DONE]"]))
                events = []
                model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder"))
                try:
                    with self.assertRaises(OutputError) as caught:
                        await model.complete("system", {}, timeout=3, on_activity=events.append)
                finally:
                    await model.close()
                self.assertEqual(caught.exception.usage["total_tokens"], 9)
                self.assertNotIn("private-", str(caught.exception))
                self.assertTrue(any(event.characters for event in events))

    async def test_empty_chunks_and_heartbeats_are_not_model_activity(self):
        self.responses.append((200, [": keep-alive", self.chunk({"role": "assistant"}),
                                    self.chunk({"content": "", "reasoning_content": ""}),
                                    self.chunk({"content": "有效文本"}), self.chunk({}, finish="stop"), "data: [DONE]"]))
        events = []
        model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder"))
        try:
            reply = await model.complete("system", {}, timeout=3, on_activity=events.append)
        finally:
            await model.close()
        self.assertIsNone(reply.total_tokens)
        self.assertEqual([(event.kind, event.characters) for event in events if event.characters], [("output", 4)])

    async def test_incomplete_stream_is_retryable_and_next_response_is_independent(self):
        self.responses.append((200, [self.chunk({"content": "must-not-prefix-next-response"}), "data: [DONE]"]))
        self.responses.append((200, [self.chunk({"content": "完整结果"}), self.chunk({}, finish="stop"), "data: [DONE]"]))
        model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder"))
        try:
            with self.assertRaises(TransientModelError) as caught:
                await model.complete("system", {}, timeout=3)
            self.assertNotIn("must-not-prefix", str(caught.exception))
            reply = await model.complete("system", {}, timeout=3)
            self.assertEqual(reply.text, "完整结果")
        finally:
            await model.close()

    async def test_only_explicit_stream_parameter_rejection_allows_fallback(self):
        errors = (
            (400, {"param": "stream", "type": "invalid_request_error", "code": "unsupported_parameter"}, True),
            (400, {"param": "model", "code": "unsupported_parameter"}, False),
            (422, {"message": "This model does not support streaming."}, True),
            (400, {"message": "Unsupported parameter: 'stream_options'."}, True),
            (400, {"message": "Streaming is not supported for this endpoint."}, True),
            (400, {"message": "当前模型不支持流式输出。"}, True),
            (400, {"message": "Unknown model: stream-model"}, False),
            (400, {"message": "Streaming connection was interrupted."}, False),
            (400, {"message": "Request failed."}, False),
            (500, {"param": "stream", "code": "unsupported_parameter"}, False),
        )
        for status, detail, unsupported in errors:
            self.responses.append((status, {"error": {"message": "do-not-print-error-body", **detail}}))
            model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder"))
            try:
                with self.assertRaises(Exception) as caught:
                    await model.complete("system", {}, timeout=3)
                self.assertEqual(isinstance(caught.exception, StreamingUnsupportedError), unsupported)
                self.assertNotIn("do-not-print", str(caught.exception))
                self.assertNotIn(detail.get("message", "do-not-print-error-body"), str(caught.exception))
            finally:
                await model.close()

    async def test_gateway_json_result_is_reused_and_cached_as_non_streaming(self):
        events = []
        model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder"))
        try:
            first = await model.complete("system", {}, timeout=3, on_activity=events.append)
            self.assertEqual(len(self.requests), 1)
            self.assertFalse(model.streaming)
            second = await model.complete("system", {}, timeout=3)
        finally:
            await model.close()
        self.assertEqual(first.text, second.text)
        self.assertTrue(self.requests[0][1]["stream"])
        self.assertFalse(self.requests[1][1]["stream"])
        self.assertNotIn("stream_options", self.requests[1][1])
        self.assertTrue(any(event.mode == "non_streaming" for event in events))

    async def test_response_reading_can_exceed_network_timeout(self):
        entered, release = threading.Event(), threading.Event()
        def delayed(handler):
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            entered.set()
            release.wait(3)
            events = [self.chunk({"content": "迟到的结果"}), self.chunk({}, finish="stop")]
            try:
                for event in events:
                    handler.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        self.responses.append((200, delayed))
        model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder", timeout=0.1))
        task = asyncio.create_task(model.complete("system", {}, timeout=0.1))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            await asyncio.sleep(0.15)
            self.assertFalse(task.done())
            network = model._client().request_timeout
            self.assertIsNone(network.read)
            self.assertEqual((network.connect, network.write, network.pool), (0.1, 0.1, 0.1))
            release.set()
            self.assertEqual((await asyncio.wait_for(task, 2)).text, "迟到的结果")
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await model.close()

    async def test_cancel_while_silent_closes_local_stream(self):
        release = threading.Event()
        connected = asyncio.Event()
        modes = 0
        def silent(handler):
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            handler.wfile.write(b": connected\n\n")
            handler.wfile.flush()
            release.wait(3)
        self.responses.append((200, silent))
        model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder"))
        raw_responses = []
        raw_client = model._client().async_client.with_raw_response
        create = raw_client.create
        async def capture(**kwargs):
            response = await create(**kwargs)
            raw_responses.append(response)
            return response
        def activity(event):
            nonlocal modes
            if event.kind == "mode":
                modes += 1
                if modes == 2:
                    connected.set()
        try:
            with patch.object(raw_client, "create", capture):
                task = asyncio.create_task(model.complete("system", {}, timeout=3, on_activity=activity))
                await asyncio.wait_for(connected.wait(), 2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(raw_responses[0].http_response.is_closed)
        finally:
            release.set()
            if "task" in locals():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await model.close()
        self.assertIsNone(model._async_http)

    async def test_physical_mid_stream_disconnect_is_retryable_and_partial_body_is_discarded(self):
        def broken(handler):
            body = ("data: " + json.dumps(self.chunk({"content": "private-partial-response"})) + "\n\n").encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Content-Length", str(len(body) + 1024))
            handler.end_headers()
            handler.wfile.write(body)
            handler.wfile.flush()
            handler.close_connection = True
        self.responses.append((200, broken))
        self.responses.append((200, [self.chunk({"content": "新的完整结果"}),
                                    self.chunk({}, finish="stop"), "data: [DONE]"]))
        model = LangChainChatModel(ModelConfig("a", "test", self.url, "offline-placeholder"))
        events = []
        try:
            with self.assertRaises(TransientModelError) as caught:
                await model.complete("system", {}, timeout=3, on_activity=events.append)
            self.assertNotIn("private-partial", str(caught.exception))
            self.assertGreater(sum(event.characters for event in events), 0)
            reply = await model.complete("system", {}, timeout=3)
            self.assertEqual(reply.text, "新的完整结果")
            self.assertIsNone(reply.total_tokens)
        finally:
            await model.close()

    async def test_loopback_stream_rejection_fallback_and_cached_calls_are_separately_accounted(self):
        self.responses.append((400, {"error": {"param": "stream", "code": "unsupported_parameter"}}))
        configs = {role: ModelConfig(role, "test", self.url, "offline-placeholder")
                   for role in ("a", "b", "judge")}
        runner = Optimizer(configs, RunOptions(retries=0))
        try:
            await runner._call("a", "system", {}, "generate-a")
            await runner._call("a", "system", {}, "repair")
            metadata = runner.metadata()
            self.assertEqual([wire[1]["stream"] for wire in self.requests], [True, False, False])
            self.assertEqual([row["call_mode"] for row in metadata["calls"]],
                             ["streaming", "non_streaming", "non_streaming"])
            self.assertEqual([row["attempt"] for row in metadata["calls"]], [1, 2, 1])
            self.assertEqual(metadata["request_count"], 3)
            self.assertEqual(metadata["known_total_tokens"], 28)
            self.assertEqual(metadata["unknown_usage_requests"], 1)
            self.assertIsNone(metadata["total_tokens"])
            self.assertEqual(metadata["calls"][1]["finish_reason"], "stop")
        finally:
            await runner.aclose()

    async def test_loopback_stream_fallback_obeys_unknown_usage_budget(self):
        self.responses.append((400, {"error": {"param": "stream_options", "code": "unsupported_parameter"}}))
        configs = {role: ModelConfig(role, "test", self.url, "offline-placeholder")
                   for role in ("a", "b", "judge")}
        runner = Optimizer(configs, RunOptions(retries=0, token_budget=100))
        try:
            with self.assertRaises(BudgetExceeded):
                await runner._call("a", "system", {}, "generate-a")
            self.assertEqual(len(self.requests), 1)
            self.assertEqual(runner.unknown_usage, 1)
        finally:
            await runner.aclose()


if __name__ == "__main__":
    unittest.main()
