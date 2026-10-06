"""Prompt-data and integration checks, not claims about real model quality."""
import io
import json
import re
import unittest
from unittest.mock import patch

import evaluate
from optimizer_config import ROOT
from optimizer_examples import GENERATION_EXAMPLES
from optimizer_layers import FOUR_LAYER_POLICY, MATERIAL_REFERENCE_POLICY, LayerAnalysis, analysis_prompt
from optimizer_models import Draft, parse_output
from optimizer_prompts import (
    QUALITY_POLICY, STRATEGIES, SYSTEM_PROMPT, clarification_prompt, generation_prompt, review_prompt,
)


def four_layer_sections(prompt):
    headings = list(re.finditer(r"(?m)^## (指令层|情境层|参考层|输出层)\n", prompt))
    return {match[1]: prompt[match.end():headings[index + 1].start() if index + 1 < len(headings) else len(prompt)]
            for index, match in enumerate(headings)}


class FourLayerPolicyTests(unittest.TestCase):
    def test_clarification_generation_repair_review_and_legacy_analysis_share_the_policy(self):
        prompts = [clarification_prompt(), review_prompt(), analysis_prompt()] + [
            generation_prompt(strategy, examples=[]) for strategy in ("a", "b", "repair")]
        for prompt in prompts:
            with self.subTest(prompt=prompt[:40]):
                self.assertIn(FOUR_LAYER_POLICY, prompt)
                self.assertIn(MATERIAL_REFERENCE_POLICY, prompt)
                self.assertIn("不为凑齐四层", prompt)
                self.assertIn("提示词本身的格式", prompt)
                self.assertIn("从 1 开始的阿拉伯数字连续序号", prompt)
                self.assertIn("每步独占一行，步骤之间用换行隔开", prompt)
                self.assertIn("简单任务不为编号而强行增加步骤", prompt)
                self.assertIn("不改变下游最终输出格式", prompt)
                self.assertIn("用户明确指定提示词本身格式时优先遵守", prompt)
                self.assertNotIn("缺失层不再默认省略", prompt)

    def test_generation_and_repair_cleanup_precedes_final_source_and_format_checks(self):
        for strategy in ("a", "b", "repair"):
            with self.subTest(strategy=strategy):
                prompt = generation_prompt(strategy, examples=[])
                self.assertIn(SYSTEM_PROMPT, prompt)
                self.assertEqual(prompt.count(QUALITY_POLICY), 1)
                self.assertLess(prompt.index("交稿前主动去冗余"),
                                prompt.index("精简与排版后，对照完整 original_request"))
                self.assertIn("同层重复、跨层复述、重复步骤和套话", prompt)
                self.assertIn("只有信息与作用都重复、合并不改变原意时才合并", prompt)
                self.assertIn("必要的任务定位、交付物定义、不同用途的复述和用户明确要求的重复强调应保留", prompt)
                self.assertIn("原样保留的参考、代码、材料及程序固定快照不作为删改对象", prompt)
                self.assertIn("指令层步骤编号与换行", prompt)
                self.assertIn("本次生成或修复中完成", prompt)
                self.assertIn("不输出中间草稿或自查过程", prompt)

    def test_multistep_examples_use_consecutive_numbered_lines(self):
        for index in (1, 4, 6, 7):
            with self.subTest(index=index):
                example = GENERATION_EXAMPLES[index]
                parsed = parse_output(json.dumps(example["output"], ensure_ascii=False), Draft)
                instructions = four_layer_sections(parsed.optimized_prompt)["指令层"]
                steps = list(re.finditer(r"(?m)^(\d+)\. (\S[^\n]*)$", instructions))
                self.assertGreaterEqual(len(steps), 2)
                self.assertEqual([int(step[1]) for step in steps], list(range(1, len(steps) + 1)))
                self.assertEqual(len(re.findall(r"\d+\.\s", instructions)), len(steps))
                self.assertIn("\n".join(step[0] for step in steps), instructions)

    def test_short_example_stays_short_and_complex_examples_omit_absent_references(self):
        simple = GENERATION_EXAMPLES[0]["output"]["optimized_prompt"]
        self.assertEqual(simple, "请解释缓存的基本概念与用途。")
        self.assertEqual(four_layer_sections(simple), {})
        for index in (0, 2, 3, 5, 8, 9, 10, 11):
            with self.subTest(single_action=index):
                prompt = GENERATION_EXAMPLES[index]["output"]["optimized_prompt"]
                instructions = four_layer_sections(prompt).get("指令层", prompt)
                self.assertNotRegex(instructions, r"(?m)^\d+\. ")
        for index in (1, 4, 6, 7):
            with self.subTest(index=index):
                sections = four_layer_sections(GENERATION_EXAMPLES[index]["output"]["optimized_prompt"])
                self.assertEqual(list(sections), ["指令层", "情境层", "输出层"])
                self.assertTrue(all(section.strip() for section in sections.values()))

    def test_agent_examples_separate_working_material_reference_and_actual_delivery(self):
        for index, variable in ((8, "{source_code}"), (9, "{notes}")):
            example = GENERATION_EXAMPLES[index]
            sections = four_layer_sections(example["output"]["optimized_prompt"])
            with self.subTest(index=index):
                self.assertEqual(list(sections), ["指令层", "情境层", "参考层", "输出层"])
                self.assertIn(variable, sections["情境层"])
                self.assertIn(example["layer_decisions"][0]["value"], sections["参考层"])
                self.assertNotIn("示例输入：", sections["参考层"])
                self.assertTrue(all(section.strip() for section in sections.values()))
        code = four_layer_sections(GENERATION_EXAMPLES[8]["output"]["optimized_prompt"])
        self.assertIn("直接修改现有文件", code["指令层"])
        self.assertIn("不新增文件", code["指令层"])
        self.assertIn("简短说明改动", code["输出层"])
        report = four_layer_sections(GENERATION_EXAMPLES[9]["output"]["optimized_prompt"])
        self.assertIn("实际保存报告到 {destination}", report["输出层"])
        self.assertIn("最后仅给文件路径", report["输出层"])

    def test_markdown_prompt_keeps_downstream_json_and_existing_outer_contract(self):
        for index in (3, 7):
            example = GENERATION_EXAMPLES[index]
            result = parse_output(json.dumps(example["output"], ensure_ascii=False), Draft)
            sections = four_layer_sections(result.optimized_prompt)
            with self.subTest(index=index):
                self.assertIn("JSON", sections["输出层"])
                self.assertIn("不要其他字段", sections["输出层"])
                self.assertFalse(result.optimized_prompt.startswith("```"))
        self.assertEqual(set(Draft.model_json_schema()["properties"]), {
            "status", "optimized_prompt", "preserved_constraints", "clarification_questions", "change_summary"})
        self.assertEqual(set(LayerAnalysis.model_json_schema()["$defs"]["Layer"]["properties"]["layer"]["enum"]), {
            "task", "context", "audience", "tone", "role", "constraints", "references", "output"})
        self.assertEqual(set(STRATEGIES), {"a", "b", "repair"})


class PromptDataTests(unittest.TestCase):
    def test_few_shot_outputs_are_valid_and_evidence_quotes_exist(self):
        for example in GENERATION_EXAMPLES:
            with self.subTest(request=example["original_request"]):
                result = parse_output(json.dumps(example["output"], ensure_ascii=False), Draft)
                sources = [example["original_request"]] + [d["value"] for d in example.get("layer_decisions", [])
                                                            if d["choice"] != "omit"]
                for constraint in result.preserved_constraints:
                    self.assertTrue(any(constraint.source_quote in source for source in sources))

    def test_selected_examples_in_demonstrations_are_preserved_in_the_prompt(self):
        chosen = [example for example in GENERATION_EXAMPLES if example.get("layer_decisions")]
        self.assertTrue(chosen)
        for example in chosen:
            for decision in example["layer_decisions"]:
                if decision["layer"] == "references" and decision["choice"] == "model":
                    self.assertIn(decision["value"], example["output"]["optimized_prompt"])

    def test_focused_evaluation_is_valid_and_separate_from_demonstrations(self):
        cases = evaluate.load_cases(ROOT / "evals" / "prompt_patterns.jsonl")
        old_cases = evaluate.load_cases(ROOT / "evals" / "cases.jsonl")
        example_inputs = {e["original_request"] for e in GENERATION_EXAMPLES}
        self.assertTrue(example_inputs.isdisjoint(c.input for c in cases + old_cases))
        self.assertTrue({c.id for c in cases}.isdisjoint(c.id for c in old_cases))
        self.assertTrue(all(c.split == "dev" for c in cases))
        self.assertTrue(any(c.expected_clarification for c in cases))
        for case in cases:
            for literal in case.required_literals:
                self.assertIn(literal, case.input)

    def test_focused_eval_dry_run_does_not_load_credentials_or_call_models(self):
        output = io.StringIO()
        with patch("evaluate.read_environment", side_effect=AssertionError("config access")), \
                patch("evaluate.Optimizer", side_effect=AssertionError("model access")), \
                patch("sys.stdout", output):
            code = evaluate.main(["--cases", str(ROOT / "evals" / "prompt_patterns.jsonl"), "--limit", "12"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result["dry_run"])
        self.assertEqual(len(result["cases"]), 12)

    def test_few_shot_steps_cases_are_valid_and_separate_from_existing_data(self):
        cases = evaluate.load_cases(ROOT / "evals" / "few_shot_steps.jsonl")
        old_cases = (evaluate.load_cases(ROOT / "evals" / "cases.jsonl")
                     + evaluate.load_cases(ROOT / "evals" / "prompt_patterns.jsonl"))
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({c.id for c in cases}), len(cases))
        self.assertTrue({c.id for c in cases}.isdisjoint(c.id for c in old_cases))
        known_inputs = {e["original_request"] for e in GENERATION_EXAMPLES} | {c.input for c in old_cases}
        self.assertTrue(known_inputs.isdisjoint(c.input for c in cases))
        self.assertTrue(all(c.split == "dev" for c in cases))
        self.assertTrue(any(c.expected_clarification for c in cases))
        for case in cases:
            for literal in case.required_literals:
                self.assertIn(literal, case.input)

    def test_few_shot_steps_dry_run_has_no_config_or_model_access(self):
        output = io.StringIO()
        with patch("evaluate.read_environment", side_effect=AssertionError("config access")), \
                patch("evaluate.Optimizer", side_effect=AssertionError("model access")), \
                patch("sys.stdout", output):
            code = evaluate.main(["--cases", str(ROOT / "evals" / "few_shot_steps.jsonl"), "--limit", "12"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result["dry_run"])
        self.assertEqual(len(result["cases"]), 12)

    def test_cot_cases_are_valid_and_separate_from_examples_and_existing_sets(self):
        cases = evaluate.load_cases(ROOT / "evals" / "cot_reasoning.jsonl")
        old_cases = [case for name in ("cases.jsonl", "prompt_patterns.jsonl", "few_shot_steps.jsonl")
                     for case in evaluate.load_cases(ROOT / "evals" / name)]
        self.assertEqual(len(cases), 12)
        self.assertTrue({c.id for c in cases}.isdisjoint(c.id for c in old_cases))
        known_inputs = {e["original_request"] for e in GENERATION_EXAMPLES} | {c.input for c in old_cases}
        self.assertTrue(known_inputs.isdisjoint(c.input for c in cases))
        self.assertTrue(all(c.split == "dev" for c in cases))
        self.assertTrue(any(c.expected_clarification for c in cases))
        for case in cases:
            for literal in case.required_literals:
                self.assertIn(literal, case.input)

    def test_cot_dry_run_does_not_load_credentials_or_call_models(self):
        output = io.StringIO()
        with patch("evaluate.read_environment", side_effect=AssertionError("config access")), \
                patch("evaluate.Optimizer", side_effect=AssertionError("model access")), \
                patch("sys.stdout", output):
            code = evaluate.main(["--cases", str(ROOT / "evals" / "cot_reasoning.jsonl"), "--limit", "12"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result["dry_run"])
        self.assertEqual(len(result["cases"]), 12)


if __name__ == "__main__":
    unittest.main()
