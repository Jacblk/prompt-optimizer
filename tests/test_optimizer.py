from __future__ import annotations

import asyncio
import importlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import legacy_optimize as optimize
from optimizer_config import ConfigurationError, ModelConfig, load_models, model_config, read_environment
from optimizer_engine import BudgetExceeded, Optimizer, OptimizationResult, RunOptions
from optimizer_layers import LAYER_LABELS
from optimizer_models import (
    Draft, ModelCallError, ModelReply, OutputError, Review, TransientModelError, parse_output)

ORIGINAL = "仅检查 src，不改文件。"


def draft(text="甲结果", **updates):
    value = {"status": "ready", "optimized_prompt": text, "preserved_constraints": [],
             "clarification_questions": [], "change_summary": []}
    value.update(updates)
    return value


def review(payload, action="select", *, selected=None):
    original_id = payload["original_candidate_id"]
    candidates = payload["candidates"]
    if selected is None:
        selected = next((c["candidate_id"] for c in candidates if c["text"].startswith("甲")),
                        next(c["candidate_id"] for c in candidates if c["candidate_id"] != original_id))
    if action == "keep_original":
        selected = original_id
    if action in {"needs_review", "needs_clarification"}:
        selected = None
    reviews = []
    evidence = payload.get("original_request", next(candidate["text"] for candidate in candidates
                                                   if candidate["candidate_id"] == original_id))
    for candidate in candidates:
        failed = action == "repair" and candidate["candidate_id"] == selected
        # A repair fixture has no passed alternative; otherwise C must select it.
        uncertain = action == "repair" and not failed
        findings = ([{"kind": "constraint_lost", "source_quote": "不改文件",
                      "candidate_quote": "", "explanation": "只读条件遗漏。"}] if failed else
                    [{"kind": "uncertain", "source_quote": evidence.strip()[:64],
                      "candidate_quote": "", "explanation": "离线修复场景未将此备选判为合格。"}] if uncertain else [])
        reviews.append({"candidate_id": candidate["candidate_id"],
                         "verdict": "fail" if failed else "uncertain" if uncertain else "pass",
                         "findings": findings,
                        "clarity": 4, "conciseness": 4, "reason": "已对照原文。"})
    return {"reviews": reviews, "action": action, "candidate_id": selected,
            "clarification_questions": ["需要检查哪些问题？"] if action == "needs_clarification" else [],
            "reason": "已检查。"}


def configs():
    return {role: ModelConfig(role, "model-" + role, "http://127.0.0.1:9/v1", "offline-placeholder")
            for role in ("a", "b", "judge")}


def reference_example(input_text, output_text):
    return {"kind": "input_output", "purpose": "演示输入与期望输出的对应关系。",
            "usage": "仅参考映射与格式，演示内容不是当前任务事实。",
            "input_text": input_text, "output_text": output_text}


def layer_analysis(original=ORIGINAL, missing=None):
    missing = missing or {}
    return {"layers": [
        {"layer": name, "status": "missing" if name in missing else "present",
         "source_quotes": [] if name in missing else [original],
         "question": (label + "有什么要求？") if name in missing else "",
         "suggestion": missing.get(name, ""), "needs_confirmation": False,
         "reference_materials": [reference_example("检查演示目录，不修改文件。",
                                                   "演示目录检查完毕，未修改文件。")]
         if name == "references" and name in missing else []}
        for name, label in LAYER_LABELS.items()
    ]}


class Scripts:
    def __init__(self, **outputs):
        self.outputs = {role: list(items) for role, items in outputs.items()}
        self.calls = []
        self.closed = []

    def factory(self, config):
        owner = self

        class FakeModel:
            def disable_streaming(self):
                self.streaming = False

            async def complete(self, system, payload, *, timeout, on_activity=None):
                owner.calls.append((config.role, system, payload, timeout))
                analyzing = payload.get("phase") == "layer_analysis"
                queue = owner.outputs.get("analysis" if analyzing else config.role, [])
                item = queue.pop(0) if queue else (layer_analysis(payload["original_request"]) if analyzing
                        else review(payload) if config.role == "judge"
                        else draft("乙结果" if config.role == "b" else "甲结果"))
                if isinstance(item, Exception):
                    raise item
                if callable(item):
                    item = item(payload)
                if asyncio.iscoroutine(item):
                    item = await item
                if isinstance(item, ModelReply):
                    return item
                return ModelReply(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False), 7, 4, 11)

            async def close(self):
                owner.closed.append(config.role)

        return FakeModel()


def engine(scripts=None, **options):
    scripts = scripts or Scripts()
    return Optimizer(configs(), RunOptions(**options), factory=scripts.factory, rng=random.Random(42))


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_generation_and_three_calls(self):
        started = set()
        both_started = asyncio.Event()

        def generator(role):
            async def generate(payload):
                started.add(role)
                if len(started) == 2:
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), timeout=1)
                return draft("甲结果" if role == "a" else "乙结果")
            return generate

        scripts = Scripts(a=[generator("a")], b=[generator("b")])
        result = await engine(scripts).run(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.optimized_prompt, "甲结果")
        self.assertEqual(result.metadata["request_count"], 3)
        self.assertEqual(result.metadata["total_tokens"], 33)
        self.assertCountEqual(scripts.closed, ["a", "b", "judge"])

    async def test_judge_gets_anonymous_candidates_and_full_original(self):
        scripts = Scripts()
        await engine(scripts).run(ORIGINAL)
        payload = scripts.calls[-1][2]
        self.assertEqual(payload["original_request"], ORIGINAL)
        for candidate in payload["candidates"]:
            self.assertEqual(set(candidate), {"candidate_id", "status", "text", "clarification_questions"})
        self.assertNotIn("model-a", json.dumps(payload))

    async def test_product_always_uses_independent_review(self):
        result = await engine().run(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertTrue(result.reviewed)
        self.assertEqual(result.metadata["request_count"], 3)

    async def test_pending_generation_is_not_final(self):
        pending = draft("待确认草稿", status="needs_clarification", clarification_questions=["检查哪些问题？"])
        scripts = Scripts(a=[pending], b=[pending], judge=[lambda p: review(p, "needs_clarification")])
        result = await engine(scripts).run(ORIGINAL)
        self.assertEqual(result.status, "needs_clarification")
        self.assertFalse(result.reviewed)

    async def test_original_can_win(self):
        result = await engine(Scripts(judge=[lambda p: review(p, "keep_original")])).run(ORIGINAL)
        self.assertEqual(result.optimized_prompt, ORIGINAL)

    async def test_clarification_from_judge(self):
        result = await engine(Scripts(judge=[lambda p: review(p, "needs_clarification")])).run(ORIGINAL)
        self.assertEqual(result.status, "needs_clarification")
        self.assertIsNone(result.optimized_prompt)

    async def test_review_runs_once_without_a_reversed_request(self):
        scripts = Scripts()
        result = await engine(scripts).run(ORIGINAL)
        self.assertEqual(result.status, "ready")
        orders = [[c["candidate_id"] for c in call[2]["candidates"]]
                  for call in scripts.calls if call[0] == "judge"]
        self.assertEqual(len(orders), 1)
        self.assertEqual(result.metadata["request_count"], 3)
        self.assertEqual([entry["purpose"] for entry in result.reviews], ["review"])

    async def test_unused_second_review_is_never_requested(self):
        scripts = Scripts(judge=[review, lambda p: review(p, "keep_original")])
        result = await engine(scripts).run(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertTrue(result.reviewed)
        self.assertEqual(len(scripts.outputs["judge"]), 1)

    async def test_repair_only_once_then_recheck(self):
        scripts = Scripts(a=[draft("甲初稿"), draft("甲修复稿")],
                          judge=[lambda p: review(p, "repair"), review])
        result = await engine(scripts).run(ORIGINAL)
        self.assertEqual(result.optimized_prompt, "甲修复稿")
        self.assertEqual(result.metadata["request_count"], 5)
        self.assertEqual(len(result.candidates), 4)
        repair_payload = next(c[2] for c in scripts.calls if "repair" in c[2])
        self.assertEqual(repair_payload["original_request"], ORIGINAL)
        self.assertTrue(repair_payload["repair"]["findings"])

    async def test_second_repair_request_stops(self):
        scripts = Scripts(judge=[lambda p: review(p, "repair"), lambda p: review(p, "repair")])
        result = await engine(scripts).run(ORIGINAL)
        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.metadata["request_count"], 5)

    async def test_repaired_result_gets_one_independent_review(self):
        scripts = Scripts(judge=[lambda p: review(p, "repair"), lambda p: review(p, "keep_original")])
        result = await engine(scripts).run(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.metadata["request_count"], 5)

    async def test_repair_can_be_disabled(self):
        result = await engine(Scripts(judge=[lambda p: review(p, "repair")]), allow_repair=False).run(ORIGINAL)
        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.metadata["request_count"], 3)

    async def test_one_generator_failure_is_visible(self):
        result = await engine(Scripts(a=[ModelCallError("a failed")])).run(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertEqual(len(result.warnings), 1)
        self.assertIsNone(result.metadata["total_tokens"])

    async def test_both_generator_failures_stop_before_judge(self):
        scripts = Scripts(a=["broken JSON"], b=[ModelCallError("b failed")])
        with self.assertRaises(OutputError):
            await engine(scripts).run(ORIGINAL)
        self.assertEqual(len(scripts.calls), 2)

    async def test_judge_failure_never_returns_unreviewed_quality_result(self):
        with self.assertRaises(OutputError):
            await engine(Scripts(judge=["not JSON"])).run(ORIGINAL)

    async def test_temporary_failure_has_counted_retry(self):
        result = await engine(Scripts(a=[TransientModelError("retry"), draft()])).run(ORIGINAL)
        self.assertEqual(result.metadata["request_count"], 4)
        self.assertIsNone(result.metadata["total_tokens"])

    async def test_retries_keep_full_request_timeout_after_long_activity(self):
        async def fail_after_long_activity(payload):
            runner.started -= 1000
            raise TransientModelError("retry")

        scripts = Scripts(a=[fail_after_long_activity, draft()])
        runner = engine(scripts)
        result = await runner.run(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.metadata["request_count"], 4)
        self.assertGreaterEqual(result.metadata["elapsed_seconds"], 1000)
        for role, _, _, timeout in scripts.calls:
            self.assertEqual(timeout, runner.configs[role].timeout)

    async def test_request_count_does_not_stop_large_runs(self):
        scripts = Scripts()

        class ManyCallsOptimizer(Optimizer):
            async def _workflow(self):
                for _ in range(65):
                    await self._call("a", "offline preparation", {"original_request": self.original}, "preparation")
                return await super()._workflow()

        runner = ManyCallsOptimizer(configs(), RunOptions(), factory=scripts.factory)
        result = await runner.run(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.metadata["request_count"], 68)
        self.assertCountEqual(scripts.closed, ["a", "b", "judge"])

    async def test_known_token_budget_stops_next_stage(self):
        runner = engine(token_budget=15)
        with self.assertRaises(BudgetExceeded):
            await runner.run(ORIGINAL)
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(runner.known_tokens, 22)  # Soft scheduling cap, not a billing cap.

    async def test_unknown_tokens_stop_budgeted_run(self):
        scripts = Scripts(a=[ModelReply(json.dumps(draft()))])
        with self.assertRaises(BudgetExceeded):
            await engine(scripts, token_budget=1000).run(ORIGINAL)
        # An immediate response can stop B before it starts; real I/O may leave B in flight.
        self.assertLessEqual(len(scripts.calls), 2)
        self.assertNotIn("judge", [call[0] for call in scripts.calls])

    async def test_failure_with_token_budget_does_not_retry(self):
        scripts = Scripts(a=[TransientModelError("unknown cost")])
        with self.assertRaises(BudgetExceeded):
            await engine(scripts, token_budget=100).run(ORIGINAL)
        self.assertEqual(len(scripts.calls), 1)

    async def test_network_timeout_still_closes_models(self):
        from dataclasses import replace
        async def slow(payload):
            raise TimeoutError("offline network timeout")
        scripts = Scripts(a=[slow], b=[slow])
        runner = engine(scripts, retries=0)
        runner.configs = {role: replace(config, timeout=0.03) for role, config in runner.configs.items()}
        with self.assertRaises(OutputError):
            await runner.run(ORIGINAL)
        self.assertCountEqual(scripts.closed, ["a", "b"])
        self.assertTrue(all(c["status"] == "transient_error" for c in runner.calls))

    async def test_invalid_constraint_quote_is_rejected(self):
        bad = draft(preserved_constraints=[{"source_quote": "不存在的条件", "constraint": "虚构"}])
        with self.assertRaises(OutputError):
            await engine(Scripts(a=[bad], b=[bad])).run(ORIGINAL)

    async def test_input_and_single_use_validation(self):
        runner = engine(max_input_chars=10)
        for invalid in ("", "  ", "长" * 11):
            with self.subTest(value=invalid), self.assertRaises(ConfigurationError):
                await runner.run(invalid)
        await runner.run("有效")
        with self.assertRaises(ConfigurationError):
            await runner.run("有效")


class SchemaTests(unittest.TestCase):
    def test_fence_is_allowed_without_extra_prose(self):
        self.assertEqual(parse_output("```json\n" + json.dumps(draft()) + "\n```", Draft).status, "ready")
        with self.assertRaises(OutputError):
            parse_output("explanation " + json.dumps(draft()), Draft)

    def test_draft_validation_rejects_extra_missing_empty_and_mismatched_fields(self):
        cases = [draft(unexpected=True), draft(optimized_prompt="  "), draft(status="needs_clarification"),
                 draft(clarification_questions=["why?"]), {"status": "ready"}, [],
                 draft(status="needs_clarification", clarification_questions=["  "])]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(OutputError):
                parse_output(json.dumps(value), Draft)

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(OutputError):
            parse_output('{"status":"ready", "status":"needs_clarification"}', Draft)

    def test_parse_errors_do_not_echo_provider_content(self):
        with self.assertRaises(OutputError) as caught:
            parse_output('{"secret":"do-not-print"}', Draft)
        self.assertNotIn("do-not-print", str(caught.exception))

    def _fixture(self):
        from optimizer_engine import Candidate
        candidates = [Candidate("orig", "original", Draft.model_validate(draft(ORIGINAL))),
                      Candidate("a", "a", Draft.model_validate(draft("甲结果")))]
        payload = {"original_candidate_id": "orig", "candidates": [c.anonymous() for c in candidates]}
        return candidates, payload

    def test_scores_are_strict_bounded_integers(self):
        _, payload = self._fixture()
        for score in (True, "5", 5.0, -1, 6):
            data = review(payload)
            data["reviews"][0]["clarity"] = score
            with self.subTest(score=score), self.assertRaises(OutputError):
                parse_output(json.dumps(data), Review)

    def test_missing_duplicate_and_unknown_candidate_ids(self):
        candidates, payload = self._fixture()
        for kind in ("missing", "duplicate", "unknown"):
            data = review(payload)
            if kind == "missing":
                data["reviews"].pop()
            elif kind == "duplicate":
                data["reviews"].append(data["reviews"][0])
            else:
                data["candidate_id"] = "absent"
            with self.subTest(kind=kind), self.assertRaises(OutputError):
                Optimizer._validate_review(Review.model_validate(data), ORIGINAL, candidates)

    def test_failed_candidate_cannot_be_selected(self):
        candidates, payload = self._fixture()
        data = review(payload, "repair")
        data["action"] = "select"
        with self.assertRaises(OutputError):
            Optimizer._validate_review(Review.model_validate(data), ORIGINAL, candidates)

    def test_clarification_candidate_cannot_be_selected(self):
        candidates, payload = self._fixture()
        candidates[-1].draft = Draft.model_validate(draft(status="needs_clarification", clarification_questions=["why?"]))
        with self.assertRaises(OutputError):
            Optimizer._validate_review(Review.model_validate(review(payload)), ORIGINAL, candidates)

    def test_fabricated_quotes_and_uncertain_repair_rejected(self):
        candidates, payload = self._fixture()
        for key, value in (("source_quote", "凭空的要求"), ("candidate_quote", "凭空的候选内容"), ("kind", "uncertain")):
            data = review(payload, "repair")
            data["reviews"][-1]["findings"][0][key] = value
            with self.subTest(key=key), self.assertRaises(OutputError):
                Optimizer._validate_review(Review.model_validate(data), ORIGINAL, candidates)

    def test_pass_requires_no_findings_and_fail_requires_evidence(self):
        _, payload = self._fixture()
        for verdict in ("pass", "uncertain"):
            data = review(payload, "repair" if verdict == "pass" else "select")
            data["reviews"][-1]["verdict"] = verdict
            with self.subTest(verdict=verdict), self.assertRaises(OutputError):
                parse_output(json.dumps(data), Review)


class ConfigTests(unittest.TestCase):
    def good(self):
        return {"MODEL_NAME": "test", "MODEL_API_KEY": "do-not-print", "MODEL_BASE_URL": "http://localhost:123/v1"}

    def test_import_has_no_dotenv_or_model_side_effects(self):
        with patch("dotenv.dotenv_values", side_effect=AssertionError("unexpected env read")), \
                patch("optimizer_models.LangChainChatModel._client", side_effect=AssertionError("unexpected client")):
            importlib.reload(optimize)

    def test_no_env_reads_process_only(self):
        with patch.dict(os.environ, {"MARKER": "process"}, clear=True), \
                patch("dotenv.dotenv_values", side_effect=AssertionError("env read")):
            self.assertEqual(read_environment(None), {"MARKER": "process"})

    def test_explicit_utf8_env_process_priority_and_no_interpolation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "settings"
            path.write_text("NAME=文件\nEMPTY=file\nRAW=${NAME}\n", encoding="utf-8-sig")
            with patch.dict(os.environ, {"NAME": "process", "EMPTY": ""}, clear=True):
                values = read_environment(path)
            self.assertEqual(values["NAME"], "process")
            self.assertEqual(values["EMPTY"], "")
            self.assertEqual(values["RAW"], "${NAME}")

    def test_config_repr_and_errors_hide_values(self):
        self.assertNotIn("do-not-print", repr(model_config(self.good(), "MODEL", "a")))
        values = self.good() | {"MODEL_EXTRA_BODY": "do-not-print"}
        with self.assertRaises(ConfigurationError) as caught:
            model_config(values, "MODEL", "a")
        self.assertNotIn("do-not-print", str(caught.exception))

    def test_invalid_connection_and_generation_parameters(self):
        for key, value in (("BASE_URL", "https://user:pass@example.com/v1"), ("BASE_URL", "https://example.com?key=secret"),
                           ("TEMPERATURE", "nan"), ("TIMEOUT", "-1"), ("MAX_TOKENS", "0"),
                           ("EXTRA_BODY", '{"messages":[]}'), ("EXTRA_BODY", "[]"),
                           ("EXTRA_BODY", '{"reasoning":NaN}'),
                           ("TOKEN_LIMIT_FIELD", "invalid"), ("JSON_MODE", "maybe")):
            with self.subTest(key=key, value=value), self.assertRaises(ConfigurationError):
                model_config(self.good() | {"MODEL_" + key: value}, "MODEL", "a")

    def test_product_loads_all_roles_and_ignores_legacy_single_model(self):
        with self.assertRaises(ConfigurationError):
            load_models(self.good())
        values = {key.replace("MODEL_", prefix + "_"): value
                  for prefix in ("GENERATOR_A", "GENERATOR_B", "JUDGE")
                  for key, value in self.good().items()}
        self.assertEqual(set(load_models(values)), {"a", "b", "judge"})
        self.assertTrue(all(config.name == "test" for config in load_models(values).values()))

    def test_quality_requires_all_roles(self):
        with self.assertRaises(ConfigurationError) as caught:
            load_models({})
        self.assertIn("GENERATOR_A_NAME", str(caught.exception))
        self.assertIn("GENERATOR_B_NAME", str(caught.exception))
        self.assertIn("JUDGE_NAME", str(caught.exception))

    def test_option_validation(self):
        for options in ({"retries": 6},
                        {"token_budget": 0}, {"max_input_chars": -1}):
            with self.subTest(options=options), self.assertRaises(ConfigurationError):
                RunOptions(**options)

    def test_removed_product_options_are_not_accepted(self):
        for options in ({"mode": "quick"}, {"strict_review": True}, {"workflow": "handoff"}):
            with self.subTest(options=options), self.assertRaises(TypeError):
                RunOptions(**options)


class CliTests(unittest.TestCase):
    def test_removed_cli_returns_utf8_tui_instruction_from_other_directory(self):
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("MODEL_", "GENERATOR_A_", "GENERATOR_B_", "JUDGE_"))}
        environment["PYTHONIOENCODING"] = "ascii"
        with tempfile.TemporaryDirectory() as temp:
            completed = subprocess.run(
                [sys.executable, str(optimize.ROOT / "optimize.py"), "--no-env", "--mode", "quality", "--check-config"],
                cwd=temp, env=environment, capture_output=True, timeout=10, check=False)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("只使用 TUI", completed.stderr.decode("utf-8"))
        self.assertEqual(completed.stdout, b"")

    def test_historical_fixture_uses_current_reviewed_pipeline(self):
        self.assertEqual(optimize.make_parser().parse_args([]).mode, "quality")

    def invoke(self, args, result=None, stdin=""):
        output, errors = io.StringIO(), io.StringIO()
        self.requests = []
        test = self

        class FakeOptimizer:
            def __init__(self, configs, options):
                pass

            async def run(self, request):
                test.requests.append(request)
                return result or OptimizationResult("ready", "最终提示词", True,
                                                     metadata={"request_count": 3, "total_tokens": 33})

        with patch("legacy_optimize.read_environment", return_value={}), patch("legacy_optimize.load_models", return_value=configs()):
            code = optimize.main(args, stdin=io.StringIO(stdin), stdout=output, stderr=errors,
                                 optimizer_factory=FakeOptimizer)
        return code, output.getvalue(), errors.getvalue()

    def test_pipe_end_is_material(self):
        code, output, _ = self.invoke(["--no-save"], stdin="before\nEND\nafter\n{payload}")
        self.assertEqual(code, 0)
        self.assertEqual(self.requests[0], "before\nEND\nafter\n{payload}")
        self.assertEqual(output, "最终提示词\n")

    def test_file_input_utf8_bom_and_end(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.txt"
            path.write_text("输入\nEND\n尾部", encoding="utf-8-sig")
            code, _, _ = self.invoke(["--no-save", "--input-file", str(path)])
        self.assertEqual(code, 0)
        self.assertEqual(self.requests[0], "输入\nEND\n尾部")

    def test_interactive_end(self):
        stream = io.StringIO("first\nEND\nignored\n")
        stream.isatty = lambda: True
        args = optimize.make_parser().parse_args([])
        self.assertEqual(optimize.read_request(args, stream, io.StringIO()), "first\n")

    def test_empty_request_exit_two(self):
        code, _, _ = self.invoke(["--request", "  ", "--no-save"])
        self.assertEqual(code, 2)
        self.assertEqual(self.requests, [])

    def test_save_failure_keeps_stdout(self):
        with patch("legacy_optimize.atomic_write", side_effect=OSError("not writable")):
            code, output, _ = self.invoke(["--request", ORIGINAL])
        self.assertEqual(code, 5)
        self.assertIn("最终提示词", output)

    def test_pending_state_does_not_overwrite_previous_result(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "last.md"
            path.write_text("existing", encoding="utf-8")
            result = OptimizationResult("needs_clarification", "草稿", False, ["范围？"])
            code, output, _ = self.invoke(["--request", ORIGINAL, "--output", str(path)], result)
            self.assertEqual(path.read_text(encoding="utf-8"), "existing")
        self.assertEqual(code, 4)
        self.assertIn("待确认", output)

    def test_default_output_is_project_relative(self):
        with tempfile.TemporaryDirectory() as temp, patch("legacy_optimize.ROOT", Path(temp)):
            code, _, _ = self.invoke(["--request", ORIGINAL])
            self.assertEqual((Path(temp) / "last_optimized_prompt.md").read_text(encoding="utf-8"), "最终提示词")
        self.assertEqual(code, 0)

    def test_json_output_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "nested" / "report.json"
            code, output, _ = self.invoke(["--request", ORIGINAL, "--no-save", "--format", "json", "--report", str(report)])
            self.assertEqual(json.loads(output), json.loads(report.read_text(encoding="utf-8")))
        self.assertEqual(code, 0)

    def test_colliding_paths_stop_before_model(self):
        code, _, _ = self.invoke(["--request", ORIGINAL, "--output", "same", "--report", "same"])
        self.assertEqual(code, 2)
        self.assertEqual(self.requests, [])

    def test_config_check_does_not_read_input_or_run_model(self):
        code, output, _ = self.invoke(["--check-config"])
        self.assertEqual(code, 0)
        self.assertEqual(self.requests, [])
        self.assertIn("尚未验证", output)

    def test_model_error_exit_three_hides_untrusted_error(self):
        with patch("legacy_optimize.read_environment", return_value={}), patch("legacy_optimize.load_models", return_value=configs()):
            errors = io.StringIO()
            code = optimize.main(["--request", ORIGINAL, "--no-save"], stdout=io.StringIO(), stderr=errors,
                                 optimizer_factory=lambda *args: (_ for _ in ()).throw(RuntimeError("do-not-print")))
        self.assertEqual(code, 3)
        self.assertNotIn("do-not-print", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
