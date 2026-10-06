import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import legacy_launcher as launcher
import legacy_optimize as optimize
from optimizer_config import ConfigurationError
from optimizer_engine import BudgetExceeded, Optimizer, RunOptions
from optimizer_layers import LayerAnalysis, LayerDecision, prompt_for_layers, validate_decisions
from optimizer_models import Draft, OutputError, Review, parse_output
from test_optimizer import ORIGINAL, Scripts, configs, draft, layer_analysis, reference_example, review


class LayerInputTests(unittest.TestCase):
    def test_each_missing_layer_has_a_real_choice_and_manual_input_can_span_lines(self):
        plan = LayerAnalysis.model_validate(layer_analysis(missing={
            "audience": "一般读者", "tone": "自然友好", "references": "一个明确标为演示的例子"}))
        output = io.StringIO()
        decisions = prompt_for_layers(plan, io.StringIO("1\n新同事\n没有相关经验\nEND\n2\n3\n"), output)
        self.assertEqual([(d.layer, d.choice, d.value) for d in decisions], [
            ("audience", "user", "新同事\n没有相关经验"),
            ("tone", "model", "自然友好"), ("references", "omit", "")])
        self.assertNotIn("缺少：任务目标", output.getvalue())

    def test_empty_answers_and_invalid_choices_never_default_to_omission(self):
        plan = LayerAnalysis.model_validate(layer_analysis(missing={"tone": "友好"}))
        output = io.StringIO()
        decisions = prompt_for_layers(plan, io.StringIO("\nwrong\n1\nEND\n3\n"), output)
        self.assertEqual(decisions[0].choice, "omit")
        self.assertIn("不会默认选择", output.getvalue())
        self.assertIn("不会自动省略", output.getvalue())

    def test_eof_and_cancellation_do_not_supply_choices(self):
        plan = LayerAnalysis.model_validate(layer_analysis(missing={"tone": "友好"}))
        for value in ("", "0\n", "1\n未完成的补充\n"):
            with self.subTest(value=value):
                self.assertIsNone(prompt_for_layers(plan, io.StringIO(value), io.StringIO()))

    def test_analysis_requires_complete_unique_layers_and_evidence(self):
        for kind in ("missing", "duplicate", "false_evidence"):
            data = layer_analysis()
            if kind == "missing":
                data["layers"].pop()
            elif kind == "duplicate":
                data["layers"][0] = data["layers"][1]
            else:
                data["layers"][0]["source_quotes"] = []
            with self.subTest(kind=kind), self.assertRaises(OutputError):
                parse_output(json.dumps(data), LayerAnalysis)

    def test_decisions_must_cover_missing_layers_and_use_the_shown_model_suggestion(self):
        plan = LayerAnalysis.model_validate(layer_analysis(missing={"tone": "友好"}))
        for choices in ([], [{"layer": "tone", "choice": "model", "value": "别的内容"}],
                        [{"layer": "tone", "choice": "omit", "value": "仍添加"}],
                        [{"layer": "role", "choice": "omit", "value": ""}]):
            with self.subTest(choices=choices), self.assertRaises(ConfigurationError):
                validate_decisions(plan, choices)

    def test_model_examples_show_and_adopt_pairs_instead_of_advisory_text(self):
        data = layer_analysis(missing={"references": "可提供示例；若不需要示例，可写“不要示例”。"})
        plan = LayerAnalysis.model_validate(data)
        output = io.StringIO()
        decisions = prompt_for_layers(plan, io.StringIO("2\n"), output)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].choice, "model")
        self.assertIn("输入：检查演示目录，不修改文件。", decisions[0].value)
        self.assertIn("输出：演示目录检查完毕，未修改文件。", decisions[0].value)
        self.assertNotIn("不要示例", decisions[0].value)
        self.assertNotIn("不要示例", output.getvalue())
        self.assertEqual(validate_decisions(plan, decisions), decisions)

    def test_missing_examples_require_nonblank_concrete_pairs(self):
        for pairs in ([], [reference_example("检查演示目录", " ")],
                      [reference_example("", "目录未修改")]):
            data = layer_analysis(missing={"references": "不要示例"})
            next(i for i in data["layers"] if i["layer"] == "references")["reference_materials"] = pairs
            with self.subTest(pairs=pairs), self.assertRaises(OutputError):
                parse_output(json.dumps(data), LayerAnalysis)

    def test_proposed_examples_reject_repeated_or_conflicting_inputs(self):
        for second_output in ("可检查", "路径缺失"):
            data = layer_analysis(missing={"references": ""})
            next(i for i in data["layers"] if i["layer"] == "references")["reference_materials"] = [
                reference_example("目录可读取", "可检查"),
                reference_example(" 目录可读取 ", second_output),
            ]
            with self.subTest(output=second_output), self.assertRaises(OutputError):
                parse_output(json.dumps(data), LayerAnalysis)

    def test_distinct_examples_may_share_the_same_valid_label(self):
        data = layer_analysis(missing={"references": ""})
        next(i for i in data["layers"] if i["layer"] == "references")["reference_materials"] = [
            reference_example("演示目录甲可读取", "可检查"),
            reference_example("演示目录乙可读取", "可检查"),
        ]
        plan = parse_output(json.dumps(data), LayerAnalysis)
        self.assertEqual(len(plan.missing()[0].reference_materials), 2)


class LayerWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def runner(self, scripts, resolver=None, **options):
        return Optimizer(configs(), RunOptions(choose_layers=True, **options),
                         factory=scripts.factory, layer_resolver=resolver)

    async def test_all_generators_and_reviews_receive_the_same_choices_and_count_analysis(self):
        plan = layer_analysis(missing={"tone": "自然友好", "references": "演示"})
        scripts = Scripts(analysis=[plan])
        choices = [LayerDecision(layer="tone", choice="user", value="温和且直接"),
                   LayerDecision(layer="references", choice="omit", value="")]

        def choose(analysis):
            self.assertEqual(len(scripts.calls), 1)
            return choices

        result = await self.runner(scripts, choose).run(ORIGINAL)
        self.assertEqual(result.metadata["request_count"], 4)
        self.assertEqual(result.metadata["calls"][0]["purpose"], "layer_analysis")
        self.assertEqual(result.layer_decisions, [c.model_dump() for c in choices])
        for _, _, payload, _ in scripts.calls[1:]:
            self.assertEqual(payload["layer_decisions"], result.layer_decisions)
            self.assertEqual(payload["original_request"], ORIGINAL)
            self.assertNotIn("suggestion", json.dumps(payload, ensure_ascii=False))

    async def test_no_resolver_or_cancellation_stops_before_generation_and_closes_model(self):
        for resolver, expected in ((None, "needs_clarification"), (lambda p: None, "cancelled")):
            scripts = Scripts(analysis=[layer_analysis(missing={"tone": "友好"})])
            result = await self.runner(scripts, resolver).run(ORIGINAL)
            self.assertEqual(result.status, expected)
            self.assertIsNone(result.optimized_prompt)
            self.assertEqual(len(scripts.calls), 1)
            self.assertEqual(scripts.closed, ["a"])

    async def test_user_wait_is_excluded_from_elapsed_statistics(self):
        scripts = Scripts(analysis=[layer_analysis(missing={"tone": "友好"})])

        def choose(plan):
            time.sleep(0.15)
            return [LayerDecision(layer="tone", choice="omit", value="")]

        result = await self.runner(scripts, choose).run(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertGreaterEqual(result.metadata["user_wait_seconds"], 0.14)
        self.assertLess(result.metadata["elapsed_seconds"], 0.1)

    async def test_source_quotes_can_use_chosen_supplements_but_not_unselected_suggestions(self):
        for quote, succeeds in (("温和且直接", True), ("热情活泼", False)):
            candidate = draft(preserved_constraints=[{"source_quote": quote, "constraint": "语气要求"}])
            scripts = Scripts(analysis=[layer_analysis(missing={"tone": "热情活泼"})],
                              a=[candidate], b=[candidate])
            runner = self.runner(scripts, lambda p: [LayerDecision(layer="tone", choice="user", value="温和且直接")])
            if succeeds:
                self.assertEqual((await runner.run(ORIGINAL)).status, "ready")
            else:
                with self.assertRaises(OutputError):
                    await runner.run(ORIGINAL)

    async def test_unknown_facts_cannot_be_published_as_ready_after_model_fill(self):
        plan = layer_analysis(missing={"context": "[待确认：输入材料所在位置]"})
        next(item for item in plan["layers"] if item["layer"] == "context")["needs_confirmation"] = True
        scripts = Scripts(analysis=[plan])
        result = await self.runner(scripts, lambda p: [LayerDecision(
            layer="context", choice="model", value=p.missing()[0].suggestion)]).run(ORIGINAL)
        self.assertEqual(result.status, "needs_clarification")
        self.assertIsNone(result.selected_id)
        self.assertFalse(result.reviewed)

    async def test_oversized_supplement_and_fabricated_analysis_quote_stop_generation(self):
        scripts = Scripts(analysis=[layer_analysis(missing={"tone": "友好"})])
        with self.assertRaises(ConfigurationError):
            await self.runner(scripts, lambda p: [LayerDecision(layer="tone", choice="user", value="长" * 30)],
                              max_input_chars=30).run(ORIGINAL)
        self.assertEqual(len(scripts.calls), 1)
        plan = layer_analysis()
        plan["layers"][0]["source_quotes"] = ["未提供的材料"]
        scripts = Scripts(analysis=[plan])
        with self.assertRaises(OutputError):
            await self.runner(scripts).run(ORIGINAL)
        self.assertEqual(len(scripts.calls), 1)

    async def test_analysis_consumes_the_same_token_budget(self):
        scripts = Scripts()
        with self.assertRaises(BudgetExceeded):
            await self.runner(scripts, token_budget=11).run(ORIGINAL)
        self.assertEqual(len(scripts.calls), 1)

    async def test_judge_cannot_select_original_that_lacks_chosen_supplements(self):
        scripts = Scripts(analysis=[layer_analysis(missing={"tone": "友好"})],
                          judge=[lambda p: review(p, "keep_original")])
        with self.assertRaises(OutputError):
            await self.runner(scripts, lambda p: [LayerDecision(layer="tone", choice="model", value="友好")]).run(ORIGINAL)

    async def test_analysis_correction_precedes_choice_and_counts_toward_usage(self):
        invalid = layer_analysis(missing={"references": "若不需要可写“不要示例”"})
        next(i for i in invalid["layers"] if i["layer"] == "references")["reference_materials"] = []
        scripts = Scripts(analysis=[invalid, layer_analysis(missing={"references": ""})])

        def choose(plan):
            self.assertEqual(len(scripts.calls), 2)
            return [LayerDecision(layer="references", choice="model", value=plan.missing()[0].suggestion)]

        def generate(payload):
            value = payload["layer_decisions"][0]["value"]
            self.assertIn("输入：", value)
            self.assertIn("输出：", value)
            self.assertNotIn("不要示例", value)
            return draft("甲：" + ORIGINAL + "\n" + value)

        scripts.outputs["a"] = [generate]
        scripts.outputs["b"] = [generate]
        result = await self.runner(scripts, choose).run(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertEqual([c["purpose"] for c in result.metadata["calls"]],
                         ["layer_analysis", "layer_analysis_repair", "generate", "generate", "review"])
        self.assertEqual(result.metadata["request_count"], 5)
        self.assertEqual(result.metadata["total_tokens"], 55)
        self.assertEqual(len(result.warnings), 1)

    async def test_invalid_analysis_stops_after_one_correction_without_asking_user(self):
        invalid = layer_analysis(missing={"references": "不要示例"})
        next(i for i in invalid["layers"] if i["layer"] == "references")["reference_materials"] = []
        scripts = Scripts(analysis=[invalid, invalid])

        def choose(plan):
            self.fail("Invalid suggestions must never be offered to the user")

        with self.assertRaises(OutputError):
            await self.runner(scripts, choose).run(ORIGINAL)
        self.assertEqual(len(scripts.calls), 2)
        self.assertEqual(scripts.closed, ["a"])

    async def test_analysis_correction_cannot_exceed_token_budget(self):
        for budget in ({"token_budget": 11},):
            invalid = layer_analysis(missing={"references": "不要示例"})
            next(i for i in invalid["layers"] if i["layer"] == "references")["reference_materials"] = []
            scripts = Scripts(analysis=[invalid])
            with self.subTest(budget=budget), self.assertRaises(BudgetExceeded):
                await self.runner(scripts, **budget).run(ORIGINAL)
            self.assertEqual(len(scripts.calls), 1)

    async def test_discarded_example_advice_cannot_be_used_as_constraint_evidence(self):
        candidate = draft(ORIGINAL + "\n不要示例。", preserved_constraints=[
            {"source_quote": "不要示例", "constraint": "禁止举例"}])
        scripts = Scripts(analysis=[layer_analysis(missing={"references": "若不需要可写“不要示例”"})],
                          a=[candidate], b=[candidate])
        with self.assertRaises(OutputError):
            await self.runner(scripts, lambda p: [LayerDecision(
                layer="references", choice="model", value=p.missing()[0].suggestion)]).run(ORIGINAL)

    async def test_omitted_examples_cannot_become_a_ban_even_when_judge_passes(self):
        for ban in ("不要示例。", "输出要求：不要提供示例。", "Do not include examples."):
            scripts = Scripts(analysis=[layer_analysis(missing={"references": ""})],
                              a=[draft("甲：" + ORIGINAL + "\n" + ban)])
            result = await self.runner(scripts, lambda p: [LayerDecision(
                layer="references", choice="omit", value="")]).run(ORIGINAL)
            with self.subTest(ban=ban):
                self.assertEqual(result.status, "needs_review")
                self.assertIsNone(result.selected_id)
                self.assertFalse(result.reviewed)
                self.assertIn(ban, result.optimized_prompt)
                self.assertIn("新增了禁止参考或禁止示例", result.reason)
                self.assertEqual(result.metadata["request_count"], 4)
                self.assertEqual(result.reviews[0]["action"], "select")

    async def test_omitting_reference_materials_does_not_restrict_downstream_answer(self):
        text = "甲：" + ORIGINAL + "\n检查结论可以按需要举例说明。"
        scripts = Scripts(analysis=[layer_analysis(missing={"references": ""})], a=[draft(text)])
        result = await self.runner(scripts, lambda p: [LayerDecision(
            layer="references", choice="omit", value="")]).run(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.optimized_prompt, text)

    async def test_explicit_prohibition_and_quoted_literals_survive_example_omission(self):
        for source in ("original", "manual", "literal"):
            original = ORIGINAL
            missing = {"references": ""}
            choices = [LayerDecision(layer="references", choice="omit", value="")]
            if source == "original":
                original += "\n无需举例。"
            elif source == "manual":
                missing["output"] = "简洁"
                choices.append(LayerDecision(layer="output", choice="user", value="请不要示例。"))
            else:
                original += "\n逐字保留这段引用：\n不要示例。"
            text = "甲：" + ORIGINAL + ("\n不要示例。" if source == "literal" else "\n不要提供示例。")
            scripts = Scripts(analysis=[layer_analysis(original, missing=missing)], a=[draft(text)])
            result = await self.runner(scripts, lambda p: choices).run(original)
            with self.subTest(source=source):
                self.assertEqual(result.status, "ready")
                self.assertEqual(result.optimized_prompt, text)

    async def test_selected_example_block_cannot_be_dropped_or_have_outputs_swapped(self):
        for change in ("drop", "swap", "replace"):
            plan = layer_analysis(missing={"references": ""})
            examples = next(i for i in plan["layers"] if i["layer"] == "references")
            examples["reference_materials"] = [
                reference_example("演示目录可读取", "可检查"),
                reference_example("演示目录不存在", "路径缺失"),
            ]

            def generate(payload):
                value = payload["layer_decisions"][0]["value"]
                if change == "drop":
                    value = ""
                elif change == "swap":
                    value = (value.replace("输出：可检查", "输出：临时标记")
                             .replace("输出：路径缺失", "输出：可检查")
                             .replace("输出：临时标记", "输出：路径缺失"))
                else:
                    value = "不要示例。"
                return draft("甲：" + ORIGINAL + "\n" + value)

            scripts = Scripts(analysis=[plan], a=[generate])
            result = await self.runner(scripts, lambda p: [LayerDecision(
                layer="references", choice="model", value=p.missing()[0].suggestion)]).run(ORIGINAL)
            with self.subTest(change=change):
                self.assertEqual(result.status, "needs_review")
                self.assertFalse(result.reviewed)
                self.assertIsNone(result.selected_id)
                self.assertIn("未完整保留", result.reason)
                self.assertEqual(result.metadata["request_count"], 4)
                if change == "swap":
                    for pair in examples["reference_materials"]:
                        self.assertIn(pair["input_text"], result.optimized_prompt)
                        self.assertIn(pair["output_text"], result.optimized_prompt)

    async def test_selected_example_block_accepts_surrounding_edits_and_windows_newlines(self):
        for newline in ("\n", "\r\n"):
            def generate(payload):
                value = payload["layer_decisions"][0]["value"].replace("\n", newline)
                return draft("甲：请完成原有检查任务。" + newline + value + newline + ORIGINAL)

            scripts = Scripts(analysis=[layer_analysis(missing={"references": ""})],
                              a=[generate])
            result = await self.runner(scripts, lambda p: [LayerDecision(
                layer="references", choice="model", value=p.missing()[0].suggestion)]).run(ORIGINAL)
            with self.subTest(newline=repr(newline)):
                self.assertEqual(result.status, "ready")
                self.assertIsNotNone(result.selected_id)


class LayerCliTests(unittest.TestCase):
    def run_cli(self, args, text, root, scripts):
        output = io.StringIO()
        with patch("legacy_optimize.ROOT", root), patch("legacy_optimize.read_environment", return_value={}), \
                patch("legacy_optimize.load_models", return_value=configs()):
            code = optimize.main(args, stdin=io.StringIO(text), stdout=output, stderr=output,
                                 optimizer_factory=lambda c, o, **extra: Optimizer(c, o, factory=scripts.factory, **extra))
        return code, output.getvalue()

    def test_cancelled_selection_saves_analysis_report_but_preserves_last_success(self):
        scripts = Scripts(analysis=[layer_analysis(missing={"tone": "友好"})])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous = root / "last_optimized_prompt.md"
            previous.write_text("上次结果", encoding="utf-8")
            report = root / "reports" / "cancel.json"
            code, output = self.run_cli(["--request", ORIGINAL, "--choose-layers", "--report", str(report)],
                                        "0\n", root, scripts)
            self.assertEqual(previous.read_text(encoding="utf-8"), "上次结果")
            data = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(code, 130)
        self.assertEqual(data["status"], "cancelled")
        self.assertEqual(data["metadata"]["request_count"], 1)
        self.assertIn("已取消", output)

    def test_choices_are_saved_and_visible_when_reopening_the_report(self):
        scripts = Scripts(analysis=[layer_analysis(missing={"tone": "自然友好", "references": "演示"})])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / "reports" / "chosen.json"
            code, _ = self.run_cli(["--request", ORIGINAL, "--choose-layers", "--report", str(report)],
                                   "2\n3\n", root, scripts)
            output = io.StringIO()
            self.assertTrue(launcher.show_report(report, output, details=True))
        self.assertEqual(code, 0)
        self.assertIn("模型补全（建议/假设）", output.getvalue())
        self.assertIn("自然友好", output.getvalue())
        self.assertIn("参考材料 — 省略", output.getvalue())
        self.assertEqual(len(scripts.calls), 4)

    def test_invented_example_ban_is_a_reported_draft_and_preserves_last_success(self):
        scripts = Scripts(analysis=[layer_analysis(missing={"references": ""})],
                          a=[draft("甲：" + ORIGINAL + "\n不要示例。")])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous = root / "last_optimized_prompt.md"
            previous.write_text("上次结果", encoding="utf-8")
            report = root / "reports" / "omitted.json"
            code, output = self.run_cli(["--request", ORIGINAL, "--choose-layers", "--report", str(report)],
                                        "3\n", root, scripts)
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(previous.read_text(encoding="utf-8"), "上次结果")
        self.assertEqual(code, 4)
        self.assertEqual(data["status"], "needs_review")
        self.assertIsNone(data["selected_id"])
        self.assertEqual(data["layer_decisions"], [{"layer": "references", "choice": "omit", "value": ""}])
        self.assertIn("需要复核", output)


if __name__ == "__main__":
    unittest.main()
