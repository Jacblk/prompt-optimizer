import asyncio
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals import cot_probe
from evals import cot_followup
from evals.live_probe import DraftBaselineOptimizer, Probe, prepare_archived_generation
from evals.review_policy_probe import CachedReviewOptimizer
from optimizer_engine import BudgetExceeded, Candidate, RunOptions
from optimizer_models import Draft, ModelActivity
from optimizer_prompts import PROMPT_VERSION
from test_optimizer import Scripts, configs


class CoTProbeTests(unittest.TestCase):
    def test_dry_run_has_no_configuration_or_model_access(self):
        output = io.StringIO()
        with patch.object(cot_probe, "read_environment", side_effect=AssertionError("configuration access")), \
                patch.object(cot_probe, "load_models", side_effect=AssertionError("model access")), \
                patch.object(cot_probe, "configure_stdio"), patch("sys.stdout", output):
            self.assertEqual(cot_probe.main([]), 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["versions"], ["2.3.0", PROMPT_VERSION])
        self.assertEqual(result["planned_requests"], 36)

    def test_baseline_restores_its_own_examples(self):
        prior = cot_probe.previous_templates()
        self.assertEqual(prior["PROMPT_VERSION"], "2.3.0")
        self.assertEqual(len(prior["GENERATION_EXAMPLES"]), 6)
        self.assertIn(json.dumps(prior["GENERATION_EXAMPLES"], ensure_ascii=False),
                      prior["generation_prompt"]("quick"))
        restored, metadata = prepare_archived_generation(prior, "a", baseline=True)
        self.assertEqual(restored, prior["generation_prompt"]("quick"))
        self.assertEqual(metadata["example_count"], 6)

    def test_downstream_checks_reject_wrong_answer_and_extra_format(self):
        self.assertTrue(cot_probe.check_response("cot-03", "结论错误，正确面积为 960 平方米。")["correct_verdict_and_area"])
        self.assertFalse(cot_probe.check_response("cot-03", "结论正确，面积为 96 平方米。")["correct_verdict_and_area"])
        self.assertTrue(cot_probe.check_response("cot-04", "UNSUPPORTED")["correct_label_and_exact_format"])
        self.assertFalse(cot_probe.check_response("cot-04", "SUPPORTED")["correct_label_and_exact_format"])
        self.assertFalse(cot_probe.check_response("cot-04", "UNSUPPORTED，因为无法推出。")["correct_label_and_exact_format"])
        self.assertTrue(cot_probe.check_response("cot-06", '{"choice":"right"}')["correct_choice_and_exact_json"])
        self.assertFalse(cot_probe.check_response("cot-06", '{"choice":"left"}')["correct_choice_and_exact_json"])
        self.assertFalse(cot_probe.check_response("cot-06", '{"choice":"right","reason":"说明"}')["correct_choice_and_exact_json"])
        self.assertFalse(cot_probe.check_response("cot-06", '{"choice":"left","choice":"right"}')["correct_choice_and_exact_json"])

    def test_followup_uses_remaining_global_budget_and_rejects_unknown_usage(self):
        data = {"status": "complete", "unknown_usage_requests": 0,
                "request_count": 35, "known_total_tokens": 153952, "elapsed_seconds": 207.62,
                "limits": {"requests": 40, "known_tokens_soft_limit": 230000, "elapsed_seconds": 1800}}
        self.assertEqual(cot_followup.remaining_limits(data)["request_limit"], 5)
        self.assertEqual(cot_followup.remaining_limits(data)["token_limit"], 76048)
        with self.assertRaises(ValueError):
            cot_followup.remaining_limits(data | {"unknown_usage_requests": 1})
        with self.assertRaises(ValueError):
            cot_followup.remaining_limits(data | {"request_count": 36})


class DevelopmentBaselineTests(unittest.IsolatedAsyncioTestCase):
    async def test_cached_policy_probe_uses_one_judge_call_without_generation_or_swap(self):
        scripts = Scripts()
        runner = CachedReviewOptimizer(configs(), RunOptions(allow_repair=False, retries=0),
                                       factory=scripts.factory)
        text = "检查代码，不改文件。"
        runner.cached = [Candidate(f"candidate_{index}", role, Draft(
            status="ready", optimized_prompt=text, preserved_constraints=[],
            clarification_questions=[], change_summary=[]))
            for index, role in enumerate(("a", "b", "original"), 1)]
        result = await runner.run(text)
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.metadata["request_count"], 1)
        self.assertEqual([call[0] for call in scripts.calls], ["judge"])
        self.assertEqual(len(result.reviews), 1)

    async def test_material_baseline_uses_single_a_call_and_no_review_backend(self):
        scripts = Scripts(a=[{"status": "ready", "optimized_prompt": "检查代码，不改文件。",
                              "preserved_constraints": [], "clarification_questions": [], "change_summary": []}])
        runner = DraftBaselineOptimizer({"a": configs()["a"]}, RunOptions(retries=0),
                                        factory=scripts.factory)
        result = await runner.run("检查代码，不改文件。")
        self.assertEqual(result.status, "unreviewed")
        self.assertEqual(result.metadata["mode"], "baseline")
        self.assertEqual(result.metadata["request_count"], 1)
        self.assertEqual([call[0] for call in scripts.calls], ["a"])
        self.assertIsNone(runner.review_backend)

    async def test_probe_quality_and_baseline_share_a_config_without_product_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            probe = Probe(Path(folder) / "probe.json", configs())
            scripts = Scripts()
            with patch.object(probe, "factory", scripts.factory), patch("sys.stdout", io.StringIO()):
                baseline = await probe.optimize("baseline", "检查代码，不改文件。", baseline=True)
                quality = await probe.optimize("quality", "检查代码，不改文件。")
            self.assertEqual(baseline["result"]["status"], "unreviewed")
            self.assertEqual(baseline["metadata"]["request_count"], 1)
            self.assertEqual(quality["result"]["status"], "ready")
            self.assertEqual(quality["metadata"]["request_count"], 3)
            self.assertEqual(set(call[0] for call in scripts.calls), {"a", "b", "judge"})

            async def slow(payload):
                await asyncio.sleep(10)

            slow_scripts = Scripts(a=[slow], b=[slow])
            with patch.object(probe, "factory", slow_scripts.factory), patch("sys.stdout", io.StringIO()):
                with self.assertRaises(BudgetExceeded):
                    await probe.optimize("probe-timeout", "检查代码，不改文件。", total_timeout=0.01)
            metadata = probe.data["records"][-1]["metadata"]
            self.assertEqual(metadata["request_count"], 2)
            self.assertTrue(all(call["status"] == "cancelled" for call in metadata["calls"]))
            self.assertCountEqual(slow_scripts.closed, ["a", "b"])

    async def test_paid_probe_wrapper_retains_wait_cap_and_activity_callback(self):
        cancelled, closed = asyncio.Event(), asyncio.Event()
        class OfflineModel:
            streaming = True
            def disable_streaming(self):
                self.streaming = False
            async def complete(self, system, payload, *, timeout, on_activity=None):
                on_activity(ModelActivity("reasoning", 4))
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            async def close(self):
                closed.set()
        raw_model = OfflineModel()
        activities = []
        with tempfile.TemporaryDirectory() as folder, patch("sys.stdout", io.StringIO()), \
                patch("evals.live_probe.LangChainChatModel", return_value=raw_model):
            probe = Probe(Path(folder) / "probe.json", configs())
            model = probe.factory(configs()["a"])
            with self.assertRaises(TimeoutError):
                await model.complete("system", {}, timeout=0.01, on_activity=activities.append)
            model.disable_streaming()
            await model.close()
            self.assertFalse(raw_model.streaming)
            self.assertTrue(cancelled.is_set())
            self.assertTrue(closed.is_set())
            self.assertEqual(activities, [ModelActivity("reasoning", 4)])
            self.assertEqual(probe.data["calls"][0]["status"], "error")
            self.assertEqual(probe.data["calls"][0]["error_type"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
