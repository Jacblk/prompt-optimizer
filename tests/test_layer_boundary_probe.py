"""Offline probe limits, source snapshots and recording contracts."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.layer_boundary_cases import CASES
from evals.layer_boundary_probe import BASELINE, BoundaryProbe, fixed_candidates, load_baseline, main, recover_reported_usage
from optimizer_engine import BudgetExceeded
from optimizer_models import ModelReply, OutputError
from test_optimizer import configs, draft


class ProbeSetupTests(unittest.TestCase):
    def test_dry_run_never_loads_config_or_constructs_models(self):
        output = io.StringIO()
        with patch("evals.layer_boundary_probe.read_environment") as load, \
             patch("evals.layer_boundary_probe.BoundaryProbe") as probe, redirect_stdout(output):
            self.assertEqual(main([]), 0)
        load.assert_not_called()
        probe.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["planned_requests"], 36)

    def test_baseline_tampering_and_changed_case_are_rejected(self):
        baseline = load_baseline(BASELINE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            for changed in ("template", "case"):
                value = json.loads(json.dumps(baseline))
                if changed == "template":
                    value["generation"][CASES[0].id]["a"]["system"] += "tampered"
                else:
                    value["cases"][0]["request"] += "changed"
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    load_baseline(path)

    def test_fixed_candidates_have_neutral_ids_and_stable_gold(self):
        for case in CASES:
            first, gold = fixed_candidates(case)
            second, next_gold = fixed_candidates(case)
            self.assertEqual([candidate.anonymous() for candidate in first],
                             [candidate.anonymous() for candidate in second])
            self.assertEqual(gold, next_gold)
            self.assertEqual(sum(value == "fail" for value in gold.values()), 1 if case.bad else 0)
            self.assertTrue(all(candidate.id.startswith("candidate_") for candidate in first))


class ProbeRecordingTests(unittest.IsolatedAsyncioTestCase):
    async def test_truncated_response_retains_known_adapter_usage(self):
        class Truncated:
            def __init__(self, config):
                pass

            async def complete(self, *args, **kwargs):
                raise OutputError("truncated", usage={"input_tokens": 5, "output_tokens": 9, "total_tokens": 14})

            async def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory, \
             patch("evals.live_probe.LangChainChatModel", Truncated), redirect_stdout(io.StringIO()):
            probe = BoundaryProbe(Path(directory) / "probe.json", configs(), load_baseline(BASELINE))
            await probe.step(CASES[0], "2.9.3", "generation", "a")
            self.assertEqual(probe.data["known_total_tokens"], 14)
            self.assertEqual(probe.data["unknown_usage_requests"], 0)
            self.assertEqual(probe.data["calls"][0]["status"], "error")

    def test_resume_recovers_numeric_usage_without_retrying_or_resetting_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            probe = BoundaryProbe(Path(directory) / "probe.json", configs(), load_baseline(BASELINE))
            data = probe.data
            data["records"] = [{"id": "old", "metadata": {"calls": [
                {"input_tokens": 5, "output_tokens": 9, "total_tokens": 14}]}}]
            data["calls"] = [{"index": 1, "test": "old", "status": "error", "total_tokens": None}]
            data["elapsed_seconds"] = 1201
            resumed = BoundaryProbe(probe.destination, configs(), probe.baseline, recover_reported_usage(data))
            self.assertEqual(resumed.data["calls"][0]["total_tokens"], 14)
            self.assertEqual(len(resumed.data["calls"]), 1)
            with self.assertRaises(BudgetExceeded):
                resumed.reserve(configs()["a"], "system", {})
            resumed.data["calls"][0]["total_tokens"] = None
            resumed.data["calls"][0]["status"] = "cancelled"
            with self.assertRaises(ValueError):
                BoundaryProbe(probe.destination, configs(), probe.baseline, resumed.data)

    async def test_mock_run_records_exactly_24_generations_and_12_reviews(self):
        calls = []

        class Transport:
            def __init__(self, config):
                self.config = config

            async def complete(self, system, payload, **kwargs):
                calls.append((self.config.role, system, payload))
                case = next(case for case in CASES if case.request == payload["original_request"])
                if self.config.role != "judge":
                    value = draft(case.good)
                else:
                    candidates = payload["candidates"]
                    assessments = []
                    for candidate in candidates:
                        failed = case.bad is not None and candidate["text"] == case.bad
                        assessments.append({"candidate_id": candidate["candidate_id"],
                                            "verdict": "fail" if failed else "pass",
                                            "findings": [{"kind": "layer_boundary" if "layer_boundary" in system else "constraint_lost",
                                                          "source_quote": case.request, "candidate_quote": case.bad,
                                                          "explanation": "Offline labelled fixture, not a semantic judge."}] if failed else [],
                                            "clarity": 4, "conciseness": 4, "reason": "offline fixture"})
                    selected = next(candidate["candidate_id"] for candidate in candidates if candidate["text"] == case.good)
                    value = {"reviews": assessments, "action": "select", "candidate_id": selected,
                             "clarification_questions": [], "reason": "offline fixture"}
                return ModelReply(json.dumps(value), 7, 4, 11, "should-not-be-recorded")

            async def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory, \
             patch("evals.live_probe.LangChainChatModel", Transport), redirect_stdout(io.StringIO()):
            destination = Path(directory) / "probe.json"
            probe = BoundaryProbe(destination, configs(), load_baseline(BASELINE))
            self.assertEqual(await probe.run(), 0)
            saved = destination.read_text(encoding="utf-8")
            self.assertNotIn("should-not-be-recorded", saved)
            self.assertNotIn("model-a", saved)
        self.assertEqual(len(calls), 36)
        self.assertEqual(sum(role == "judge" for role, _, _ in calls), 12)
        self.assertEqual(probe.data["known_total_tokens"], 396)
        self.assertTrue(all(record["metadata"]["request_count"] == 1 for record in probe.data["records"]))
        for index, case in enumerate(CASES):
            versions = [record["version"] for record in probe.data["records"] if record["case_id"] == case.id]
            self.assertEqual(versions[:2], ["2.9.2", "2.9.3"] if index % 2 == 0 else ["2.9.3", "2.9.2"])

    async def test_unknown_usage_stops_after_first_transport_request(self):
        class UnknownUsage:
            def __init__(self, config):
                pass

            async def complete(self, system, payload, **kwargs):
                return ModelReply(json.dumps(draft(CASES[0].good)))

            async def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory, \
             patch("evals.live_probe.LangChainChatModel", UnknownUsage), redirect_stdout(io.StringIO()):
            probe = BoundaryProbe(Path(directory) / "probe.json", configs(), load_baseline(BASELINE))
            self.assertEqual(await probe.run(), 4)
        self.assertEqual(probe.data["status"], "stopped_by_limit")
        self.assertEqual(probe.data["request_count"], 1)
        self.assertEqual(probe.data["unknown_usage_requests"], 1)

    async def test_all_global_limits_stop_before_another_request(self):
        for limit in ("requests", "known_tokens_soft_limit", "elapsed_seconds"):
            with tempfile.TemporaryDirectory() as directory:
                probe = BoundaryProbe(Path(directory) / "probe.json", configs(), load_baseline(BASELINE))
                probe.data["limits"][limit] = 0
                with self.subTest(limit=limit), self.assertRaises(BudgetExceeded):
                    probe.reserve(configs()["a"], "system", {"original_request": "fixture"})
                self.assertEqual(probe.data["calls"], [])
