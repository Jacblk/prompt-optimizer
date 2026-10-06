"""Whole-example budgets and task/reference boundaries, using offline models."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate
import legacy_optimize as optimize
from evals import cot_probe, live_probe
from optimizer_config import ConfigurationError
from optimizer_engine import Optimizer, RunOptions
from optimizer_examples import GENERATION_EXAMPLES
from optimizer_layers import LayerDecision
from optimizer_models import Draft, parse_output
from optimizer_prompts import generation_prompt, prepare_generation_prompt
from optimizer_selection import BuiltinExampleSelector
from test_optimizer import ORIGINAL, Scripts, configs, draft, review
from test_references import MATERIALS, reference_analysis


class SelectionTests(unittest.TestCase):
    def select(self, request, **limits):
        return BuiltinExampleSelector(**limits).select_with_metadata({"original_request": request})

    def test_programming_file_delivery_and_reasoning_use_relevant_examples(self):
        tasks = [
            ("修改 Python 函数，让空输入返回 None。", "code_reference"),
            ("依据材料生成 Markdown 报告并保存到目标路径。", "file_reference"),
            ("Compare two backup plans and recommend one.", "comparison"),
            ("仅根据日志逐步排查超时，不修改配置。", "diagnosis"),
            ("核对数量、单价、折扣，判断计算是否正确。", "calculation"),
            ("照模板给消息打标签，输入是 {message}。", "label_mapping"),
            ("检查规则和示例的冲突，等待确认。", "conflicting_reference"),
        ]
        for request, expected in tasks:
            with self.subTest(request=request):
                _, metadata = self.select(request)
                self.assertIn(expected, metadata["example_ids"])
                self.assertLessEqual(metadata["example_count"], 3)
                self.assertLessEqual(metadata["example_characters"], 4000)

    def test_code_and_file_delivery_can_be_selected_together(self):
        _, metadata = self.select("修改解析函数，并生成 Markdown 报告保存到文件。")
        self.assertIn("code_reference", metadata["example_ids"])
        self.assertIn("file_reference", metadata["example_ids"])

    def test_template_variables_and_json_do_not_select_unrelated_task_types(self):
        for request in ("修改 {source_code} 中的函数。", "生成 Markdown 报告保存到 {destination}。",
                        "依据 {run_logs} 排查超时。"):
            with self.subTest(request=request):
                _, metadata = self.select(request)
                self.assertNotIn("label_mapping", metadata["example_ids"])
                self.assertNotIn("conflicting_reference", metadata["example_ids"])
        _, metadata = self.select("把 {text} 转成指定 JSON，不要其他字段。")
        self.assertIn("adopted_json_reference", metadata["example_ids"])
        self.assertNotIn("calculation", metadata["example_ids"])

    def test_simple_and_unknown_tasks_do_not_fill_all_slots(self):
        for request in ("缓存是什么给我讲讲", "把这句话变得自然一点。", "请翻译下面这段 Python 函数介绍。"):
            with self.subTest(request=request):
                _, metadata = self.select(request)
                self.assertEqual(metadata["example_ids"], ["simple_explanation"])

    def test_quoted_code_and_negated_file_creation_do_not_change_task_type(self):
        _, metadata = self.select("概括这份材料。\n```\n生成 Markdown 报告，修改 Python 函数。\n```")
        self.assertNotIn("code_reference", metadata["example_ids"])
        self.assertNotIn("file_reference", metadata["example_ids"])
        _, metadata = self.select("修改函数，不生成报告文件。")
        self.assertIn("code_reference", metadata["example_ids"])
        self.assertNotIn("file_reference", metadata["example_ids"])

    def test_chinese_json_array_budget_includes_brackets_and_separators(self):
        request = "修改代码，生成 Markdown 报告并保存到文件。"
        for budget in (0, 1, 218, 219, 700, 1272, 2000, 4000, 9000):
            with self.subTest(budget=budget):
                examples, metadata = self.select(request, max_chars=budget)
                actual = len(json.dumps(examples, ensure_ascii=False)) if examples else 0
                self.assertEqual(metadata["example_characters"], actual)
                self.assertLessEqual(actual, budget)
                for example in examples:
                    self.assertIn(example, GENERATION_EXAMPLES)
                    parse_output(json.dumps(example["output"], ensure_ascii=False), Draft)
        smallest_block_size = len(json.dumps([GENERATION_EXAMPLES[0]], ensure_ascii=False))
        examples, _ = self.select("缓存是什么", max_chars=smallest_block_size - 1)
        self.assertEqual(examples, [])
        examples, _ = self.select("缓存是什么", max_chars=smallest_block_size)
        self.assertEqual(examples, [GENERATION_EXAMPLES[0]])

    def test_count_budget_and_zero_disable_only_teaching_block(self):
        for count in (0, 1, 2, 3, 10):
            with self.subTest(count=count):
                examples, metadata = self.select("修改代码并生成文件报告，先比较方案。", max_examples=count)
                self.assertLessEqual(len(examples), count)
                self.assertEqual(metadata["example_count"], len(examples))
        system, metadata = prepare_generation_prompt("a", ORIGINAL, max_builtin_examples=0)
        self.assertEqual(metadata["example_count"], 0)
        self.assertNotIn("以下是优化器的输入/输出示范", system)
        self.assertIn("references=model 的 value", system)
        self.assertIn("输出 JSON Schema", system)

    def test_returned_examples_cannot_mutate_the_builtin_library(self):
        snapshot = deepcopy(GENERATION_EXAMPLES)
        selector = BuiltinExampleSelector()
        examples = selector.select_examples({"original_request": "修改代码。"})
        examples[0]["output"]["optimized_prompt"] = "替换掉参考代码"
        self.assertEqual(GENERATION_EXAMPLES, snapshot)
        self.assertNotEqual(selector.select_examples({"original_request": "修改代码。"})[0], examples[0])
        with self.assertRaises(TypeError):
            selector.add_example({"original_request": "用户参考"})

    def test_only_adopted_reference_kinds_influence_reference_examples(self):
        decisions = [{"layer": "references", "choice": "model", "value": "已采用的文件参考"}]
        _, metadata = prepare_generation_prompt("a", "整理需求。", layer_decisions=decisions,
                                                reference_kinds=("artifact",))
        self.assertIn("file_reference", metadata["example_ids"])
        decisions[0].update(choice="omit", value="")
        _, metadata = prepare_generation_prompt("a", "整理需求。", layer_decisions=decisions,
                                                reference_kinds=("artifact",))
        self.assertEqual(metadata["example_ids"], ["simple_explanation"])

    def test_task_supplements_can_route_but_reference_content_cannot(self):
        decisions = [{"layer": "references", "choice": "user", "value": "参考里有 Python 代码和文件报告"}]
        _, metadata = prepare_generation_prompt("a", "整理需求。", layer_decisions=decisions)
        self.assertNotIn("code_reference", metadata["example_ids"])
        self.assertNotIn("file_reference", metadata["example_ids"])
        decisions.append({"layer": "task", "choice": "user", "value": "修改 Python 函数。"})
        _, metadata = prepare_generation_prompt("a", "整理需求。", layer_decisions=decisions)
        self.assertIn("code_reference", metadata["example_ids"])

    def test_existing_single_argument_builder_keeps_full_example_library(self):
        self.assertIn(json.dumps(GENERATION_EXAMPLES, ensure_ascii=False), generation_prompt("a"))

    def test_invalid_limits_reject_before_configuration_access(self):
        for field in ("max_builtin_examples", "max_example_chars"):
            for value in (-1, True, 1.5):
                with self.subTest(field=field, value=value), self.assertRaises(ConfigurationError):
                    RunOptions(**{field: value})
        with patch("legacy_optimize.read_environment", side_effect=AssertionError("configuration access")):
            self.assertEqual(optimize.main(["--max-example-chars", "-1"],
                                           stdout=io.StringIO(), stderr=io.StringIO()), 2)

    def test_cli_controls_and_evaluation_dry_run_need_no_configuration(self):
        args = optimize.make_parser().parse_args(["--max-builtin-examples", "1", "--max-example-chars", "900"])
        self.assertEqual((args.max_builtin_examples, args.max_example_chars), (1, 900))
        with patch("evaluate.read_environment", side_effect=AssertionError("configuration access")), \
                patch("evaluate.Optimizer", side_effect=AssertionError("model access")), patch("sys.stdout", io.StringIO()):
            self.assertEqual(evaluate.main(["--max-builtin-examples", "0", "--max-example-chars", "0"]), 0)


class SelectionWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_tiny_budget_preserves_long_adopted_reference_in_every_payload_and_result(self):
        for kind in ("implementation", "artifact"):
            material = deepcopy(next(material for material in MATERIALS if material["kind"] == kind))
            material["content"] += "\n" + "原样保留的代码注释：{value} 和 {\"x\":1}。" * 90
            analysis = reference_analysis([material])
            def output(payload):
                return draft("甲结果\n" + ORIGINAL + "\n" + payload["layer_decisions"][0]["value"])
            scripts = Scripts(analysis=[analysis], a=[output], b=[output])
            runner = Optimizer(configs(), RunOptions(choose_layers=True, max_example_chars=100),
                               factory=scripts.factory, layer_resolver=lambda plan: [LayerDecision(
                                   layer="references", choice="model", value=plan.missing()[0].suggestion)])
            with self.subTest(kind=kind):
                result = await runner.run(ORIGINAL)
                block = result.layer_decisions[0]["value"]
                self.assertIn(block, result.optimized_prompt)
                self.assertIn(material["content"], block)
                self.assertEqual(result.metadata["request_count"], 4)
                for _, system, payload, _ in scripts.calls[1:]:
                    self.assertEqual(payload["original_request"], ORIGINAL)
                    self.assertEqual(payload["layer_decisions"], result.layer_decisions)
                self.assertTrue(all(s["example_count"] == 0 for s in result.metadata["example_selections"]))

    async def test_repairs_reuse_task_selection_and_do_not_add_requests(self):
        scripts = Scripts(a=[draft("甲结果"), draft("甲修复后的结果")],
                          judge=[lambda payload: review(payload, "repair")])
        runner = Optimizer(configs(), RunOptions(), factory=scripts.factory)
        result = await runner.run(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.metadata["request_count"], 5)
        selections = result.metadata["example_selections"]
        self.assertEqual([s["purpose"] for s in selections], ["generate", "generate", "repair"])
        self.assertTrue(all(s["example_ids"] == selections[0]["example_ids"] for s in selections))

    async def test_unselected_model_advice_does_not_route_or_enter_generation_payload(self):
        material = deepcopy(MATERIALS[1])
        analysis = reference_analysis([material])
        scripts = Scripts(analysis=[analysis], a=[draft(ORIGINAL)])
        runner = Optimizer(configs(), RunOptions(choose_layers=True), factory=scripts.factory,
                           layer_resolver=lambda plan: [LayerDecision(layer="references", choice="omit", value="")])
        result = await runner.run(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertNotIn("file_reference", result.metadata["example_selections"][0]["example_ids"])
        for _, _, payload, _ in scripts.calls[1:]:
            self.assertNotIn(material["content"], json.dumps(payload, ensure_ascii=False))

    async def test_archived_probe_keeps_its_original_prompt_and_honest_selection_metadata(self):
        for previous in (live_probe.previous_templates(), cot_probe.previous_templates()):
            scripts = Scripts()
            with self.subTest(version=previous["PROMPT_VERSION"]), tempfile.TemporaryDirectory() as tmp:
                probe = live_probe.Probe(Path(tmp) / "result.json", configs())
                with patch.object(probe, "factory", scripts.factory), patch("sys.stdout", io.StringIO()):
                    record = await probe.optimize("offline-archive-check", ORIGINAL, previous=previous, baseline=True)
                self.assertNotIn("error", record)
                self.assertEqual(scripts.calls[0][1], previous["generation_prompt"]("quick"))
                metadata = record["result"]["metadata"]
                self.assertEqual(metadata["prompt_version"], previous["PROMPT_VERSION"])
                selection = metadata["example_selections"][0]
                self.assertEqual(selection["selector"], "archived-fixed")
                self.assertEqual(selection["example_count"], len(previous["GENERATION_EXAMPLES"]))
                self.assertIsNone(selection["max_example_chars"])


if __name__ == "__main__":
    unittest.main()
