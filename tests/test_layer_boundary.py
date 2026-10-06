"""Boundary evidence and repair contracts; scripted judges do not prove model quality."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evals.layer_boundary_cases import CASES
from optimizer_documents import ReferenceFile, prepare_references
from optimizer_engine import Candidate, Optimizer
from optimizer_layers import LAYER_BOUNDARY_CHECKLIST
from optimizer_models import Draft, Finding, OutputError, Review, parse_output
from optimizer_prompts import PROMPT_VERSION, generation_prompt, review_prompt
from test_optimizer import Scripts, draft, engine
from test_review_quality import judge_response, quality_case


BOUNDARY_CASES = tuple(quality_case(
    case.id, case.request, case.bad, case.good, "layer_boundary", case.request,
    "待处理输入、事实依据或交付约定实质错放；按用途归位，保留全部内容和约束。",
    candidate_quote=case.bad,
) for case in CASES if case.bad is not None)


class BoundaryEvidenceTests(unittest.TestCase):
    def fixture(self):
        case = BOUNDARY_CASES[0]
        candidates = [Candidate("a", "a", Draft.model_validate(draft(case.bad))),
                      Candidate("b", "b", Draft.model_validate(draft(case.bad))),
                      Candidate("original", "original", Draft.model_validate(draft(case.original)))]
        payload = {"original_request": case.original, "original_candidate_id": "original",
                   "candidates": [candidate.anonymous() for candidate in candidates]}
        return case, candidates, judge_response(case, payload, "repair")

    def test_boundary_kind_keeps_fields_and_legacy_missing_content_evidence(self):
        self.assertEqual(parse_output(json.dumps(BOUNDARY_CASES[0].finding), Finding).kind, "layer_boundary")
        legacy = {"kind": "constraint_lost", "source_quote": "只读", "candidate_quote": "",
                  "explanation": "恢复只读约束。"}
        self.assertEqual(parse_output(json.dumps(legacy), Finding).model_dump(), legacy)
        self.assertEqual(set(Finding.model_json_schema()["properties"]),
                         {"kind", "source_quote", "candidate_quote", "explanation"})
        self.assertEqual(set(Review.model_json_schema()["properties"]),
                         {"reviews", "action", "candidate_id", "clarification_questions", "reason"})

    def test_boundary_requires_nonblank_candidate_excerpt(self):
        for quote in ("", " \t\r\n"):
            with self.subTest(quote=repr(quote)), self.assertRaises(OutputError):
                parse_output(json.dumps(BOUNDARY_CASES[0].finding | {"candidate_quote": quote}), Finding)

    def test_boundary_source_and_candidate_quotes_must_both_exist(self):
        for key in ("source_quote", "candidate_quote"):
            case, candidates, data = self.fixture()
            finding = next(item for item in data["reviews"] if item["candidate_id"] == data["candidate_id"])["findings"][0]
            finding[key] = "not-a-real-quote"
            with self.subTest(key=key), self.assertRaises(OutputError):
                Optimizer._validate_review(Review.model_validate(data), case.original, candidates)

    def test_high_scores_do_not_release_boundary_failure(self):
        case, candidates, data = self.fixture()
        data["action"] = "select"
        for assessment in data["reviews"]:
            assessment.update(clarity=5, conciseness=5)
        with self.assertRaises(OutputError):
            Optimizer._validate_review(Review.model_validate(data), case.original, candidates)

    def test_checkpoint_follows_demonstrations_and_precedes_schema(self):
        for prompt in [review_prompt()] + [generation_prompt(strategy) for strategy in ("a", "b", "repair")]:
            with self.subTest(prompt=prompt[:30]):
                self.assertEqual(prompt.count(LAYER_BOUNDARY_CHECKLIST), 1)
                self.assertLess(prompt.index(LAYER_BOUNDARY_CHECKLIST), prompt.index("输出 JSON Schema："))
                if "以下是优化器的输入/输出示范" in prompt:
                    self.assertLess(prompt.index("以下是优化器的输入/输出示范"), prompt.index(LAYER_BOUNDARY_CHECKLIST))
        self.assertEqual(PROMPT_VERSION, "2.9.3")


class BoundaryWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_five_mixed_boundary_cases_repair_once_and_recheck(self):
        for case in BOUNDARY_CASES:
            with self.subTest(case=case.name):
                def repair(payload):
                    self.assertEqual(payload["original_request"], case.original)
                    self.assertEqual(payload["repair"], {"candidate": case.bad, "findings": [case.finding]})
                    return draft(case.fixed)
                scripts = Scripts(a=[draft(case.bad), repair], b=[draft(case.bad), repair],
                                  judge=[lambda p: judge_response(case, p, "repair"),
                                         lambda p: judge_response(case, p, "select")])
                result = await engine(scripts).run(case.original)
                self.assertEqual((result.status, result.optimized_prompt), ("ready", case.fixed))
                self.assertEqual(result.metadata["request_count"], 5)
                self.assertEqual([review["purpose"] for review in result.reviews], ["review", "repair_review"])
                self.assertTrue(all(system.count(LAYER_BOUNDARY_CHECKLIST) == 1 for _, system, _, _ in scripts.calls))

    async def test_valid_alternative_avoids_repair(self):
        case = BOUNDARY_CASES[0]
        scripts = Scripts(a=[draft(case.bad)], b=[draft(case.fixed)],
                          judge=[lambda p: judge_response(case, p, "select")])
        result = await engine(scripts).run(case.original)
        self.assertEqual((result.optimized_prompt, result.metadata["request_count"]), (case.fixed, 3))
        self.assertFalse(any("repair" in payload for _, _, payload, _ in scripts.calls))

    async def test_disabled_or_still_failing_repair_returns_needs_review(self):
        case = BOUNDARY_CASES[0]
        for enabled in (False, True):
            scripts = Scripts(a=[draft(case.bad), draft(case.bad)], b=[draft(case.bad), draft(case.bad)],
                              judge=[lambda p: judge_response(case, p, "repair"),
                                     lambda p: judge_response(case, p, "repair")])
            result = await engine(scripts, allow_repair=enabled).run(case.original)
            with self.subTest(enabled=enabled):
                self.assertEqual(result.status, "needs_review")
                self.assertIsNone(result.optimized_prompt)
                self.assertEqual(result.metadata["request_count"], 5 if enabled else 3)

    async def test_simple_original_can_be_kept_without_headings(self):
        case = CASES[-1]
        quality = quality_case("simple", case.request, "bad", case.good, "layer_boundary", case.request, "fixture")
        scripts = Scripts(a=[draft(case.good)], b=[draft(case.good)],
                          judge=[lambda p: judge_response(quality, p, "keep_original", original_pass=True)])
        result = await engine(scripts).run(case.request)
        self.assertEqual((result.optimized_prompt, result.metadata["request_count"]), (case.request, 3))

    async def test_dual_purpose_and_explicit_prompt_format_pass(self):
        examples = [(CASES[3].request, CASES[3].good),
                    ("将上述需求写成一段话，不用标题：根据{notes}生成报告，仅参考{layout}的排版，交付Markdown。",
                     "根据{notes}生成报告，事实以笔记为准；仅参考{layout}的排版，交付Markdown。")]
        for original, text in examples:
            quality = quality_case("positive", original, "bad", text, "layer_boundary", original, "fixture")
            scripts = Scripts(a=[draft(text)], b=[draft(text)],
                              judge=[lambda p: judge_response(quality, p, "select", original_pass=True)])
            result = await engine(scripts).run(original)
            self.assertEqual((result.optimized_prompt, result.metadata["request_count"]), (text, 3))
        self.assertEqual(CASES[3].good.count("{handbook}"), 1)

    async def test_fixed_reference_with_its_own_headings_survives_boundary_repair(self):
        case = BOUNDARY_CASES[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.md"
            content = "## 情境层\n这是规范原文的内部标题，不是当前提示词的归属。\n"
            path.write_text(content, encoding="utf-8")
            bundle = prepare_references([ReferenceFile(path)])
            scripts = Scripts(a=[draft(case.bad), draft(case.fixed)], b=[draft(case.bad), draft(case.fixed)],
                              judge=[lambda p: judge_response(case, p, "repair"),
                                     lambda p: judge_response(case, p, "select")])
            runner = engine(scripts)
            runner.references = bundle
            result = await runner.run(case.original)
            block = bundle.payload()[0]["block"]
            self.assertEqual(result.status, "ready")
            self.assertEqual(result.optimized_prompt.count(block), 1)
            self.assertIn(content, result.optimized_prompt)
            for _, _, payload, _ in scripts.calls:
                self.assertEqual(payload["reference_files"], bundle.payload())
