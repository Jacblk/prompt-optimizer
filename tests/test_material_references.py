"""Natural material bindings: selection and offline pipeline contracts.

Scripted model responses exercise transport and layer decisions, not real model
understanding. The separate evaluation cases require semantic review.
"""
import io
import json
import unittest
from unittest.mock import patch

import evaluate
from optimizer_config import ROOT
from optimizer_engine import Optimizer, RunOptions
from optimizer_examples import GENERATION_EXAMPLES
from optimizer_layers import FOUR_LAYER_POLICY, LayerAnalysis, MATERIAL_REFERENCE_POLICY, prompt_for_layers
from optimizer_models import OutputError
from optimizer_selection import BuiltinExampleSelector
from test_optimizer import Scripts, configs, draft, layer_analysis, review


def material_plan(request, *deferred_layers, missing=None):
    data = layer_analysis(request, missing=missing)
    for item in data["layers"]:
        if item["layer"] in deferred_layers:
            item["status"] = "deferred"
    return data


class MaterialSelectionTests(unittest.TestCase):
    def test_natural_bindings_select_both_clear_goal_and_missing_goal_examples(self):
        requests = ["根据所给视频总结观点。", "参考上述图片的布局。", "根据上图定位按钮。",
                    "按照附件模板排版。", "转写已上传的录音。", "根据所给的两个视频比较操作。", "比较这两张截图。",
                    "概括 https://example.org/notes?id=7#part 的内容。",
                    "Summarize the provided video."]
        selector = BuiltinExampleSelector()
        for request in requests:
            with self.subTest(request=request):
                _, metadata = selector.select_with_metadata({"original_request": request})
                self.assertIn("natural_material_task", metadata["example_ids"])
                self.assertIn("natural_material_goal_missing", metadata["example_ids"])
                self.assertLessEqual(metadata["example_characters"], 4000)

    def test_media_creation_or_generic_media_mentions_do_not_imply_bound_material(self):
        for request in ("制作一个视频。", "画一张图片。", "解释什么是音频。", "给定预算后制定计划。"):
            with self.subTest(request=request):
                _, metadata = BuiltinExampleSelector().select_with_metadata({"original_request": request})
                self.assertNotIn("external_material", metadata["matched_features"])

    def test_translation_and_quoted_code_do_not_route_as_external_material_tasks(self):
        for request in ('请翻译“根据所给视频”成英语。', '概括材料。\n```\n根据所给视频，总结观点。\n```'):
            with self.subTest(request=request):
                _, metadata = BuiltinExampleSelector().select_with_metadata({"original_request": request})
                self.assertNotIn("external_material", metadata["matched_features"])

    def test_whole_natural_material_examples_obey_count_and_character_budgets(self):
        for count, budget in ((0, 4000), (1, 4000), (3, 1), (3, 600), (3, 1000), (3, 4000)):
            with self.subTest(count=count, budget=budget):
                examples, metadata = BuiltinExampleSelector(max_examples=count, max_chars=budget).select_with_metadata(
                    {"original_request": "根据附件内容生成报告。"})
                self.assertLessEqual(len(examples), count)
                self.assertLessEqual(metadata["example_characters"], budget)
                for example in examples:
                    self.assertIn(example, GENERATION_EXAMPLES)


class MaterialLayerWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def runner(self, scripts, *, resolver=None, **options):
        return Optimizer(configs(), RunOptions(choose_layers=True, **options),
                         factory=scripts.factory, layer_resolver=resolver)

    async def test_unloaded_material_bindings_do_not_trigger_choices_or_implicit_loading(self):
        for request, layer in (("根据所给视频，总结主要观点。", "context"),
                               ("参考上述图片的布局设计页面。", "references"),
                               ("概括 https://example.org/guide?q=1#steps 的内容。", "context")):
            scripts = Scripts(analysis=[material_plan(request, layer)], a=[draft(request)])
            with self.subTest(request=request), patch("optimizer_documents.LocalReferenceLoader",
                    side_effect=AssertionError("implicit material loading")):
                result = await self.runner(scripts).run(request)
            self.assertEqual(result.status, "ready")
            self.assertEqual(result.layer_decisions, [])
            self.assertEqual(result.metadata["request_count"], 4)
            self.assertEqual(next(item for item in result.layers if item["layer"] == layer)["status"], "deferred")
            for _, system, payload, _ in scripts.calls:
                self.assertIn(MATERIAL_REFERENCE_POLICY, system)
                self.assertIn(FOUR_LAYER_POLICY, system)
                self.assertEqual(payload["original_request"], request)
                self.assertNotIn("reference_files", payload)

    async def test_input_and_style_bindings_remain_distinct_and_only_other_layers_are_offered(self):
        request = "内容依据附件甲，排版参考附件乙，生成一份报告。"
        data = material_plan(request, "context", "references", missing={"tone": "简洁"})
        plan = LayerAnalysis.model_validate(data)
        output = io.StringIO()
        choices = prompt_for_layers(plan, io.StringIO("3\n"), output)
        self.assertEqual([choice.layer for choice in choices], ["tone"])
        self.assertNotIn("缺少：背景与输入材料", output.getvalue())
        self.assertNotIn("缺少：参考材料", output.getvalue())
        scripts = Scripts(analysis=[data], a=[draft("甲" + request)], b=[draft("乙" + request)])
        result = await self.runner(scripts, resolver=lambda _: choices).run(request)
        self.assertEqual(result.status, "ready")
        for _, system, payload, _ in scripts.calls:
            self.assertIn(MATERIAL_REFERENCE_POLICY, system)
            self.assertEqual(payload["original_request"], request)
        self.assertIn("内容依据附件甲", result.optimized_prompt)
        self.assertIn("排版参考附件乙", result.optimized_prompt)

    async def test_material_only_request_still_requires_a_task_goal(self):
        request = "根据所给视频"
        data = material_plan(request, "context", missing={"task": "[待确认：希望基于视频完成什么任务]"})
        goal = next(item for item in data["layers"] if item["layer"] == "task")
        goal["question"] = "希望基于视频完成什么任务？"
        goal["needs_confirmation"] = True
        scripts = Scripts(analysis=[data])
        result = await self.runner(scripts).run(request)
        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual(result.questions, [goal["question"]])
        self.assertIsNone(result.optimized_prompt)
        self.assertEqual(len(scripts.calls), 1)
        self.assertEqual([item["layer"] for item in result.layers if item["status"] == "missing"], ["task"])

    async def test_real_material_mapping_ambiguity_is_not_hidden_by_deferred_status(self):
        request = "根据所给的两个视频，概括其中那个视频的操作步骤。"
        data = material_plan(request, missing={"context": "[待确认：需处理两个视频中的哪一个]"})
        binding = next(item for item in data["layers"] if item["layer"] == "context")
        binding["question"] = "需处理两个视频中的哪一个？"
        scripts = Scripts(analysis=[data])
        result = await self.runner(scripts).run(request)
        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual(result.questions, [binding["question"]])
        self.assertEqual(len(scripts.calls), 1)

    async def test_deferred_material_evidence_must_still_exist_in_the_original(self):
        request = "根据所给视频总结观点。"
        data = material_plan(request, "context")
        next(item for item in data["layers"] if item["layer"] == "context")["source_quotes"] = ["所给图片"]
        scripts = Scripts(analysis=[data])
        with self.assertRaises(OutputError):
            await self.runner(scripts).run(request)
        self.assertEqual(len(scripts.calls), 1)

    async def test_repair_and_review_share_the_material_policy_and_original_binding(self):
        request = "根据所给视频，列出操作步骤。"

        def repair_review(payload):
            result = review(payload, "repair")
            for item in result["reviews"]:
                for finding in item["findings"]:
                    finding.update(source_quote="所给视频", explanation="遗漏原有视频绑定。")
            return result

        scripts = Scripts(a=[draft("甲请列出操作步骤。"), draft("甲" + request)],
                          b=[draft("乙" + request)], judge=[repair_review])
        runner = Optimizer(configs(), RunOptions(), factory=scripts.factory)
        result = await runner.run(request)
        self.assertEqual(result.status, "ready")
        self.assertIn("所给视频", result.optimized_prompt)
        for _, system, payload, _ in scripts.calls:
            self.assertIn(MATERIAL_REFERENCE_POLICY, system)
            self.assertIn(FOUR_LAYER_POLICY, system)
            self.assertEqual(payload["original_request"], request)
        self.assertEqual(result.metadata["request_count"], 5)
        self.assertEqual(len(result.reviews), 2)
        self.assertTrue(all("reversed" not in review["purpose"] for review in result.reviews))


class MaterialEvaluationDataTests(unittest.TestCase):
    def test_evaluation_cases_are_valid_unique_and_separate_from_teaching_examples(self):
        cases = evaluate.load_cases(ROOT / "evals" / "material_references.jsonl")
        old_cases = [case for name in ("cases.jsonl", "prompt_patterns.jsonl", "few_shot_steps.jsonl", "cot_reasoning.jsonl")
                     for case in evaluate.load_cases(ROOT / "evals" / name)]
        self.assertEqual(len(cases), 14)
        self.assertTrue({case.id for case in cases}.isdisjoint(case.id for case in old_cases))
        known_inputs = {example["original_request"] for example in GENERATION_EXAMPLES} | {case.input for case in old_cases}
        self.assertTrue(known_inputs.isdisjoint(case.input for case in cases))
        self.assertTrue(any(case.expected_clarification for case in cases))
        self.assertTrue(any(not case.expected_clarification for case in cases))
        for case in cases:
            self.assertTrue(all(literal in case.input for literal in case.required_literals))

    def test_evaluation_dry_run_does_not_load_configuration_or_call_models(self):
        output = io.StringIO()
        with patch("evaluate.read_environment", side_effect=AssertionError("configuration access")), \
                patch("evaluate.Optimizer", side_effect=AssertionError("model access")), patch("sys.stdout", output):
            code = evaluate.main(["--cases", str(ROOT / "evals" / "material_references.jsonl"), "--limit", "14"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result["dry_run"])
        self.assertEqual(len(result["cases"]), 14)


if __name__ == "__main__":
    unittest.main()
