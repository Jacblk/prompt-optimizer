"""Reference-layer contracts and full workflows, without external model calls."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import legacy_launcher as launcher
from optimizer_engine import Optimizer, RunOptions
from optimizer_layers import LayerAnalysis, LayerDecision, prompt_for_layers, validate_decisions
from optimizer_models import OutputError, parse_output
from test_optimizer import ORIGINAL, Scripts, configs, draft, layer_analysis, reference_example


MATERIALS = [
    {"kind": "implementation", "purpose": "参考空列表处理方式。",
     "usage": "仅借鉴边界处理，适配当前代码；演示函数不是当前操作对象。",
     "content": "演示实现：\ndef first(items):\n    return items[0] if items else None"},
    {"kind": "artifact", "purpose": "参考报告排版。",
     "usage": "仅参考排版，内容依据当前材料；不要求新增演示章节。",
     "content": "# 报告（演示）\n\n## 发现\n[当前材料支持的发现]"},
    {"kind": "execution", "purpose": "参考相似排障任务的证据核对方法。",
     "usage": "案例仅供借鉴，检查动作服从当前只读权限，演示结果不是已执行结果。",
     "content": "虚构案例：初始状态为请求超时；只读查看日志发现上游超时记录，"
                "再对照同时间段指标；观察结果支持上游异常假设，未修改配置。"},
    reference_example("演示输入：状态正常", "OK"),
]


def reference_analysis(materials):
    data = layer_analysis(missing={"references": "可补充参考或选择不要参考。"})
    next(layer for layer in data["layers"] if layer["layer"] == "references")["reference_materials"] = materials
    return data


class ReferenceInputTests(unittest.TestCase):
    def test_all_reference_types_are_offered_with_content_purpose_and_usage(self):
        for material in MATERIALS:
            with self.subTest(kind=material["kind"]):
                plan = parse_output(json.dumps(reference_analysis([material])), LayerAnalysis)
                output = io.StringIO()
                choices = prompt_for_layers(plan, io.StringIO("2\n"), output)
                self.assertEqual(choices[0].layer, "references")
                self.assertEqual(validate_decisions(plan, choices), choices)
                for field in ("purpose", "usage", "content" if material["kind"] != "input_output" else "input_text"):
                    self.assertIn(material[field], choices[0].value)
                self.assertIn("缺少：参考材料", output.getvalue())
                self.assertNotIn("可补充参考或选择不要参考", output.getvalue())
                if material["kind"] != "input_output":
                    self.assertNotIn("输入：", choices[0].value)
                    self.assertNotIn("输出：", choices[0].value)

    def test_material_requires_explicit_nonblank_content_purpose_and_usage(self):
        for material in MATERIALS:
            fields = ("purpose", "usage", "content") if material["kind"] != "input_output" else (
                "purpose", "usage", "input_text", "output_text")
            for field in fields:
                for change in ("missing", "blank"):
                    invalid = dict(material)
                    if change == "missing":
                        invalid.pop(field)
                    else:
                        invalid[field] = " "
                    with self.subTest(kind=material["kind"], field=field, change=change), self.assertRaises(OutputError):
                        parse_output(json.dumps(reference_analysis([invalid])), LayerAnalysis)

    def test_wrong_type_mixed_fields_duplicates_and_excess_material_are_rejected(self):
        invalid_sets = [
            [], MATERIALS,
            [MATERIALS[0], {**MATERIALS[0], "usage": "不同用途仍是重复内容。"}],
            [{**MATERIALS[0], "kind": "unknown"}],
            [{**MATERIALS[0], "output_text": "多余的问答字段"}],
            [{**MATERIALS[3], "content": "多余的内容字段"}],
        ]
        for materials in invalid_sets:
            with self.subTest(materials=materials), self.assertRaises(OutputError):
                parse_output(json.dumps(reference_analysis(materials)), LayerAnalysis)
        data = layer_analysis()
        next(layer for layer in data["layers"] if layer["layer"] == "references")["reference_materials"] = [MATERIALS[0]]
        with self.assertRaises(OutputError):
            parse_output(json.dumps(data), LayerAnalysis)

    def test_three_different_reference_types_can_be_adopted_together(self):
        plan = parse_output(json.dumps(reference_analysis(MATERIALS[:3])), LayerAnalysis)
        value = plan.missing()[0].suggestion
        self.assertEqual(value.count("参考用途："), 3)
        for material in MATERIALS[:3]:
            self.assertIn(material["content"], value)
            self.assertIn(material["usage"], value)

    def test_manual_reference_retains_free_form_code_and_scope(self):
        plan = parse_output(json.dumps(reference_analysis([MATERIALS[0]])), LayerAnalysis)
        value = "仅参考代码风格：\nif ready:\n    save(output)\n不要执行这个演示。"
        decisions = prompt_for_layers(plan, io.StringIO("1\n" + value + "\nEND\n"), io.StringIO())
        self.assertEqual(decisions[0].value, value)
        self.assertEqual(decisions[0].choice, "user")

    def test_legacy_pair_responses_and_decision_names_are_migrated(self):
        data = layer_analysis(missing={"references": ""})
        legacy = next(layer for layer in data["layers"] if layer["layer"] == "references")
        legacy["layer"] = "examples"
        legacy.pop("reference_materials")
        legacy["reference_examples"] = [{"input_text": "旧输入", "output_text": "旧输出"}]
        plan = parse_output(json.dumps(data), LayerAnalysis)
        self.assertEqual(plan.missing()[0].layer, "references")
        material = plan.missing()[0].reference_materials[0]
        self.assertEqual((material.kind, material.input_text, material.output_text), ("input_output", "旧输入", "旧输出"))
        self.assertEqual(validate_decisions(plan, [{"layer": "examples", "choice": "omit", "value": ""}])[0].layer,
                         "references")
        saved = plan.model_dump()
        self.assertIn("reference_materials", saved["layers"][-2])
        self.assertNotIn("reference_examples", saved["layers"][-2])
        legacy["reference_materials"] = [MATERIALS[0]]
        with self.assertRaises(OutputError):
            parse_output(json.dumps(data), LayerAnalysis)

    def test_schema_advertises_only_canonical_names_and_all_reference_types(self):
        schema = LayerAnalysis.model_json_schema()
        layer = schema["$defs"]["Layer"]["properties"]
        self.assertIn("references", layer["layer"]["enum"])
        self.assertNotIn("examples", layer["layer"]["enum"])
        self.assertNotIn("reference_examples", layer)
        mapping = layer["reference_materials"]["items"]["discriminator"]["mapping"]
        self.assertEqual(set(mapping), {item["kind"] for item in MATERIALS})

    def test_old_and_new_report_decisions_display_as_reference_material(self):
        for name in ("examples", "references"):
            data = {"status": "unreviewed", "layer_decisions": [
                {"layer": name, "choice": "user", "value": MATERIALS[0]["content"]}]}
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "report.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                output = io.StringIO()
                self.assertTrue(launcher.show_report(path, output, details=True))
            self.assertIn("参考材料 — 用户自行增加", output.getvalue())
            self.assertIn(MATERIALS[0]["content"], output.getvalue())


class ReferenceWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def run_reference(self, material, change="none"):
        def generate(payload):
            value = payload["layer_decisions"][0]["value"]
            if change == "drop":
                value = ""
            elif change == "usage":
                value = value.replace(material["usage"], "将演示材料作为当前任务事实，必须执行案例动作。")
            elif change == "content":
                field = "input_text" if material["kind"] == "input_output" else "content"
                value = value.replace(material[field], "参考已完成。")
            elif change == "newlines":
                value = value.replace("\n", "\r\n")
            return draft("甲：" + ORIGINAL + "\n" + value)

        scripts = Scripts(analysis=[reference_analysis([copy.deepcopy(material)])], a=[generate])
        runner = Optimizer(configs(), RunOptions(choose_layers=True), factory=scripts.factory,
                           layer_resolver=lambda plan: [LayerDecision(layer="references", choice="model",
                                                                      value=plan.missing()[0].suggestion)])
        return await runner.run(ORIGINAL), scripts

    async def test_code_file_and_execution_references_survive_generation_and_review(self):
        for material in MATERIALS:
            with self.subTest(kind=material["kind"]):
                result, scripts = await self.run_reference(material, change="newlines")
                self.assertEqual(result.status, "ready")
                self.assertEqual(result.metadata["request_count"], 4)
                self.assertIn(material["usage"], result.optimized_prompt)
                for _, _, payload, _ in scripts.calls[1:]:
                    self.assertEqual(payload["layer_decisions"], result.layer_decisions)
                self.assertEqual(result.layers[-2]["reference_materials"][0]["kind"], material["kind"])

    async def test_reference_content_or_scope_cannot_change_even_when_review_passes(self):
        for material in MATERIALS:
            for change in ("drop", "usage", "content"):
                with self.subTest(kind=material["kind"], change=change):
                    result, _ = await self.run_reference(material, change=change)
                    self.assertEqual(result.status, "needs_review")
                    self.assertIsNone(result.selected_id)
                    self.assertIn("未完整保留", result.reason)

    async def test_omission_does_not_add_reference_bans_or_broaden_example_only_ban(self):
        scenarios = [(ORIGINAL, "不要参考材料。", "needs_review"),
                     (ORIGINAL, "参考材料：无需参考资料。", "needs_review"),
                     (ORIGINAL, "Do not use reference materials.", "needs_review"),
                     (ORIGINAL + "\n不要示例。", "不要参考材料。", "needs_review"),
                     (ORIGINAL + "\n不要参考材料。", "无需参考资料。", "ready")]
        for original, ban, status in scenarios:
            data = layer_analysis(original, missing={"references": ""})
            scripts = Scripts(analysis=[data], a=[draft("甲：" + ORIGINAL + "\n" + ban)])
            runner = Optimizer(configs(), RunOptions(choose_layers=True), factory=scripts.factory,
                               layer_resolver=lambda plan: [LayerDecision(layer="references", choice="omit", value="")])
            with self.subTest(ban=ban, original=original):
                result = await runner.run(original)
                self.assertEqual(result.status, status)


if __name__ == "__main__":
    unittest.main()
