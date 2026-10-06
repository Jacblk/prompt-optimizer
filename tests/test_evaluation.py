import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate
from optimizer_config import ConfigurationError, ROOT
from optimizer_engine import RunOptions
from optimizer_models import ModelCallError, OutputError, TransientModelError
from test_optimizer import Scripts, configs


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    def cases(self, limit=3):
        return evaluate.load_cases(ROOT / "evals" / "cases.jsonl")[:limit]

    async def test_global_request_cap_and_unreviewed_human_fields(self):
        scripts = Scripts()
        result = await evaluate.run_evaluation(self.cases(), configs(), RunOptions(), 4,
                         runner_factory=lambda c, o, **budget: evaluate.EvaluationOptimizer(c, o, factory=scripts.factory, **budget))
        self.assertEqual(result["cases_completed"], 1)
        self.assertEqual(result["cases_skipped_for_budget"], 2)
        self.assertEqual(result["request_count"], 3)
        self.assertIsNone(result["records"][0]["manual_review"]["intent_preserved"])
        self.assertEqual(result["records"][0]["observations"]["semantic_quality"], "requires_human_review")

    async def test_failed_calls_count_against_shared_budget(self):
        scripts = Scripts(a=[ModelCallError("offline failure"), "基线输出"])
        result = await evaluate.run_evaluation(self.cases(), {"a": configs()["a"]}, RunOptions(retries=0), 2,
                         runner_factory=lambda c, o, **budget: evaluate.BaselineOptimizer(c, o, factory=scripts.factory, **budget), baseline=True)
        self.assertEqual(result["request_count"], 2)
        self.assertEqual(result["cases_completed"], 2)
        self.assertIn("error", result["records"][0])
        self.assertIn("result", result["records"][1])

    async def test_evaluation_budget_still_counts_retries_and_parallel_attempts(self):
        scripts = Scripts(a=[TransientModelError("retry")])
        result = await evaluate.run_evaluation(self.cases(), configs(), RunOptions(), 3,
                     runner_factory=lambda c, o, **budget: evaluate.EvaluationOptimizer(
                         c, o, factory=scripts.factory, **budget))
        self.assertEqual(result["request_count"], 3)
        self.assertEqual(len(scripts.calls), 3)
        self.assertEqual(result["cases_skipped_for_budget"], 2)
        self.assertEqual(result["records"][0]["error"]["type"], "BudgetExceeded")
        self.assertNotIn("judge", [call[0] for call in scripts.calls])

    async def test_baseline_uses_archived_system_template_without_json_contract(self):
        scripts = Scripts(a=["旧版模板的输出"])
        result = await evaluate.run_evaluation(self.cases(1), {"a": configs()["a"]}, RunOptions(), 1,
                       runner_factory=lambda c, o, **budget: evaluate.BaselineOptimizer(c, o, factory=scripts.factory, **budget), baseline=True)
        record = result["records"][0]
        self.assertEqual(record["result"]["optimized_prompt"], "旧版模板的输出")
        self.assertEqual(record["metadata"]["mode"], "baseline")
        self.assertEqual(record["metadata"]["prompt_version"], "baseline-v1")
        self.assertIsNone(record["observations"]["clarification_matches_expectation"])
        self.assertEqual([call[0] for call in scripts.calls], ["a"])
        self.assertEqual(scripts.calls[0][1], (ROOT / "baselines" / "system_prompt_v1.txt").read_text(encoding="utf-8"))

    async def test_checkpoints_preserve_each_completed_case(self):
        checkpoints = []
        scripts = Scripts(a=["第一份基线", "第二份基线"])
        result = await evaluate.run_evaluation(self.cases(2), {"a": configs()["a"]}, RunOptions(), 2,
                     runner_factory=lambda c, o, **budget: evaluate.BaselineOptimizer(c, o, factory=scripts.factory, **budget),
                     checkpoint=lambda records, used: checkpoints.append((len(records), used)), baseline=True)
        self.assertEqual(checkpoints, [(1, 1), (2, 2)])
        self.assertEqual(result["cases_completed"], 2)

    async def test_invalid_evaluation_budgets(self):
        for call_budget in (0, -1, True, 1.5):
            with self.subTest(budget=call_budget), self.assertRaises(ConfigurationError):
                await evaluate.run_evaluation(self.cases(), configs(), RunOptions(), call_budget)


class EvaluationCliTests(unittest.TestCase):
    def test_baseline_dry_run_is_one_request_and_removed_options_are_rejected(self):
        output = io.StringIO()
        with patch("evaluate.read_environment", side_effect=AssertionError("env access")), redirect_stdout(output):
            self.assertEqual(evaluate.main(["--mode", "baseline", "--limit", "1", "--call-budget", "1"]), 0)
        self.assertEqual(json.loads(output.getvalue())["minimum_requests_without_failures"], 1)
        for arguments in (["--mode", "quick"], ["--strict-review"], ["--max-requests", "1"], ["--timeout", "1"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluate.main(arguments)

    def test_dry_run_never_reads_env_or_constructs_runner(self):
        output = io.StringIO()
        with patch("evaluate.read_environment", side_effect=AssertionError("env access")), \
                patch("evaluate.EvaluationOptimizer", side_effect=AssertionError("model access")), redirect_stdout(output):
            code = evaluate.main(["--split", "all", "--limit", "20"])
        self.assertEqual(code, 0)
        data = json.loads(output.getvalue())
        self.assertTrue(data["dry_run"])
        self.assertEqual(len(data["cases"]), 20)
        self.assertEqual(data["minimum_requests_without_failures"], 60)

    def test_sample_set_has_unique_ids_and_separate_holdout(self):
        cases = evaluate.load_cases(ROOT / "evals" / "cases.jsonl")
        self.assertEqual(len(cases), 20)
        self.assertEqual(sum(c.split == "holdout" for c in cases), 6)
        self.assertTrue(all(c.manual_checks for c in cases))

    def test_duplicate_ids_and_invalid_schema_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.jsonl"
            case = evaluate.load_cases(ROOT / "evals" / "cases.jsonl")[0].model_dump_json()
            path.write_text(case + "\n" + case, encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                evaluate.load_cases(path)
            path.write_text('{"id":"incomplete"}', encoding="utf-8")
            with self.assertRaises(OutputError):
                evaluate.load_cases(path)

    def test_impossible_budget_rejected_without_model_call(self):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(evaluate.main(["--call-budget", "1"]), 2)


if __name__ == "__main__":
    unittest.main()
