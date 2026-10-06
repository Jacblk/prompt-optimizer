"""Exercise the installed official evaluator without external models or .env."""
import asyncio
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langsmith.run_helpers import get_tracing_context
from openevals.llm import create_async_llm_as_judge

from optimizer_config import ConfigurationError, ModelConfig
from optimizer_engine import Optimizer, RunOptions
from optimizer_models import ModelCallError, ModelReply, OutputError, Review
from optimizer_review import evaluate_review
from evals import review_probe
from evals import review_policy_probe


def valid_review():
    return {"reviews": [{"candidate_id": "candidate_1", "verdict": "pass", "findings": [],
                         "clarity": 4, "conciseness": 4, "reason": "已检查。"}],
            "action": "keep_original", "candidate_id": "candidate_1",
            "clarification_questions": [], "reason": "保留原文。"}


class OfficialReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_official_factory_keeps_one_request_schema_and_literal_messages(self):
        calls = []
        system = '遵守 Review；示例 {"action":"select"}，保留 {tone}。'
        payload = {"original_request": '核对 {body} 和 {{literal}}。\nEND',
                   "original_candidate_id": "candidate_1", "candidates": []}

        async def request(received_system, received_payload):
            calls.append((received_system, received_payload))
            return ModelReply(json.dumps(valid_review(), ensure_ascii=False), 10, 4, 14)

        with patch("openevals.llm.create_async_llm_as_judge", wraps=create_async_llm_as_judge) as factory:
            result = await evaluate_review(system, payload, request)
        self.assertEqual(calls, [(system, payload)])
        self.assertEqual(result.model_dump(), valid_review())
        factory.assert_called_once()
        self.assertIs(factory.call_args.kwargs["output_schema"], Review)
        self.assertIsInstance(factory.call_args.kwargs["judge"], BaseChatModel)
        self.assertNotIn("model", factory.call_args.kwargs)

    async def test_strict_parser_rejects_duplicate_keys_and_coerced_scores(self):
        secret = "provider-details-must-not-appear"
        malformed = json.dumps(valid_review())[:-1] + ', "reason": "' + secret + '"}'
        coerced = valid_review()
        coerced["reviews"][0]["clarity"] = "4"
        for response in (malformed, json.dumps(coerced), secret):
            with self.subTest(response_type=response[:12]):
                async def request(system, payload):
                    return ModelReply(response)
                with self.assertRaises(OutputError) as caught:
                    await evaluate_review("system", {}, request)
                self.assertNotIn(secret, str(caught.exception))

    async def test_sdk_error_is_preserved_without_an_evaluator_retry(self):
        calls = 0
        error = ModelCallError("服务暂时不可用。")
        async def request(system, payload):
            nonlocal calls
            calls += 1
            raise error
        with self.assertRaises(ModelCallError) as caught:
            await evaluate_review("system", {}, request)
        self.assertIs(caught.exception, error)
        self.assertEqual(calls, 1)

    async def test_cancellation_reaches_the_budgeted_request(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def request(system, payload):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        task = asyncio.create_task(evaluate_review("system", {}, request))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(cancelled.is_set())
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_official_evaluator_disables_tracing_even_when_environment_enables_it(self):
        async def request(system, payload):
            self.assertIs(get_tracing_context()["enabled"], False)
            return ModelReply(json.dumps(valid_review()))
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true"}), \
                patch("langsmith.client.Client.request_with_retries", side_effect=AssertionError("trace upload")) as network:
            result = await evaluate_review("system", {}, request)
        self.assertEqual(result.action, "keep_original")
        network.assert_not_called()

    async def test_missing_backend_fails_before_any_generation_request(self):
        configs = {role: ModelConfig(role, "offline", "http://127.0.0.1:9/v1", "placeholder")
                   for role in ("a", "b", "judge")}
        runner = Optimizer(configs, RunOptions(), factory=lambda config: self.fail("model created"))
        with patch("optimizer_engine.require_review_backend", side_effect=ConfigurationError("missing dependency")):
            with self.assertRaises(ConfigurationError):
                await runner.run("检查材料。")
        self.assertEqual(runner.calls, [])


class ReviewProbeTests(unittest.TestCase):
    def test_dry_run_never_reads_configuration_or_constructs_a_model(self):
        output = io.StringIO()
        with patch.object(review_probe, "read_environment", side_effect=AssertionError("configuration read")), \
                patch.object(review_probe, "load_models", side_effect=AssertionError("model configuration")), \
                patch.object(review_probe, "ReviewProbe", side_effect=AssertionError("model probe")), \
                patch.object(review_probe, "configure_stdio"), patch("sys.stdout", output):
            self.assertEqual(review_probe.main([]), 0)
        value = json.loads(output.getvalue())
        self.assertTrue(value["dry_run"])
        self.assertEqual(value["planned_requests"], 9)
        self.assertEqual(value["roles"], ["a", "b", "judge"])

    def test_selected_followup_has_six_requests_and_does_not_load_configuration(self):
        output = io.StringIO()
        with patch.object(review_probe, "read_environment", side_effect=AssertionError("configuration read")), \
                patch.object(review_probe, "configure_stdio"), patch("sys.stdout", output):
            self.assertEqual(review_probe.main(["--cases", "cot-10", "cot-06", "--judge-timeout", "180",
                                               "--workflow-timeout", "450"]), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["cases"], ["cot-10", "cot-06"])
        self.assertEqual(value["planned_requests"], 6)

    def test_cached_review_dry_run_uses_report_material_only(self):
        original = "按规则输出标签，不添加思维链。"
        candidates = [{"id": "candidate_" + str(index), "origin": origin, "status": "ready",
                       "optimized_prompt": original, "preserved_constraints": [],
                       "clarification_questions": [], "change_summary": []}
                      for index, origin in enumerate(("a", "b", "original"), 1)]
        data = {"records": [{"id": "cot-10/quality", "input": original,
                              "result": {"candidates": candidates}}]}
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "saved-report.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(review_policy_probe, "read_environment", side_effect=AssertionError("configuration read")), \
                    patch.object(review_policy_probe, "load_models", side_effect=AssertionError("model configuration")), \
                    patch.object(review_policy_probe, "configure_stdio"), patch("sys.stdout", output):
                self.assertEqual(review_policy_probe.main(["--source-report", str(source)]), 0)
        value = json.loads(output.getvalue())
        self.assertTrue(value["dry_run"])
        self.assertEqual(value["planned_requests"], 1)
        self.assertEqual(value["generation_requests"], 0)


if __name__ == "__main__":
    unittest.main()
