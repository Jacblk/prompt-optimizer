"""Exercise actual archived layer models against the current reference layer."""
import copy
import json
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import legacy_launcher as launcher
from optimizer_config import ConfigurationError
from optimizer_layers import (
    Layer, LayerAnalysis, ReferenceExample, adds_example_ban_after_omission,
    changed_model_example_block, validate_decisions,
)
from test_optimizer import ORIGINAL


def archived_layers():
    path = Path(__file__).resolve().parents[1] / "baselines" / "optimizer_layers_v2_4_0.py.txt"
    module = types.ModuleType("archived_optimizer_layers")
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(path.read_text(encoding="utf-8-sig"), str(path), "exec"), module.__dict__)
    return module


def archived_plan(module):
    return module.LayerAnalysis.model_validate({"layers": [
        {"layer": name, "status": "missing" if name == "examples" else "present",
         "source_quotes": [] if name == "examples" else [ORIGINAL],
         "question": "提供参考示例？" if name == "examples" else "",
         "suggestion": "", "needs_confirmation": False,
         "reference_examples": [{"input_text": "演示输入甲", "output_text": "演示输出甲"},
                                {"input_text": "演示输入乙", "output_text": "演示输出乙"}]
         if name == "examples" else []}
        for name in module.LAYER_LABELS
    ]})


class LegacyCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.previous = archived_layers()
        self.previous_plan = archived_plan(self.previous)
        self.old_value = self.previous_plan.missing()[0].suggestion

    def test_legacy_example_constructor_and_layer_read_access(self):
        pair = ReferenceExample(input_text="旧输入", output_text="旧输出")
        self.assertEqual(pair.model_dump(), {"input_text": "旧输入", "output_text": "旧输出"})
        layer = Layer(layer="examples", status="missing", source_quotes=[], question="提供示例？",
                      suggestion="", needs_confirmation=False, reference_examples=[pair])
        self.assertEqual(layer.layer, "references")
        self.assertEqual(layer.reference_examples[0].model_dump(), pair.model_dump())
        self.assertEqual(layer.reference_materials[0].kind, "input_output")

    def test_archived_analysis_json_and_model_selection_can_be_reused(self):
        for choice in ("model", "user", "omit"):
            value = self.old_value if choice == "model" else "旧的手动参考" if choice == "user" else ""
            old = self.previous.LayerDecision(layer="examples", choice=choice, value=value)
            analysis = LayerAnalysis.model_validate_json(self.previous_plan.model_dump_json())
            with self.subTest(choice=choice):
                for _ in range(3):
                    decisions = validate_decisions(analysis, [json.loads(old.model_dump_json())])
                    self.assertEqual(decisions[0].layer, "references")
                    self.assertEqual(decisions[0].value, value)
                    analysis = LayerAnalysis.model_validate_json(analysis.model_dump_json())

    def test_archived_model_objects_can_supply_decisions(self):
        analysis = LayerAnalysis.model_validate(self.previous_plan.model_dump())
        old = self.previous.LayerDecision(layer="examples", choice="model", value=self.old_value)
        self.assertEqual(validate_decisions(analysis, [old])[0].value, self.old_value)

    def test_legacy_helper_names_protect_archived_decision_objects(self):
        old = self.previous.LayerDecision(layer="examples", choice="model", value=self.old_value)
        self.assertFalse(changed_model_example_block(ORIGINAL + "\n" + self.old_value, [old]))
        self.assertTrue(changed_model_example_block(ORIGINAL, [old]))
        omit = self.previous.LayerDecision(layer="examples", choice="omit", value="")
        self.assertTrue(adds_example_ban_after_omission(ORIGINAL + "\n不要示例。", ORIGINAL, [omit]))

    def test_legacy_model_selection_cannot_swap_or_drop_outputs(self):
        analysis = LayerAnalysis.model_validate(self.previous_plan.model_dump())
        bad_values = [self.old_value.replace("演示输出甲", "演示输出乙"),
                      self.old_value.replace("输出：演示输出甲", ""), self.old_value + "\n新增要求。"]
        for value in bad_values:
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                validate_decisions(analysis, [{"layer": "examples", "choice": "model", "value": value}])

    def test_legacy_format_cannot_discard_usage_from_modern_references(self):
        data = copy.deepcopy(self.previous_plan.model_dump())
        legacy = next(layer for layer in data["layers"] if layer["layer"] == "examples")
        pairs = legacy.pop("reference_examples")
        legacy["layer"] = "references"
        legacy["reference_materials"] = [{"kind": "input_output", "purpose": "参考当前格式。",
            "usage": "必须保留这个参考范围，不得将案例当成当前任务事实。", **pair} for pair in pairs]
        analysis = LayerAnalysis.model_validate(data)
        with self.assertRaises(ConfigurationError):
            validate_decisions(analysis, [{"layer": "examples", "choice": "model", "value": self.old_value}])

    def test_early_advisory_only_report_can_be_viewed_but_not_replayed(self):
        data = copy.deepcopy(self.previous_plan.model_dump())
        legacy = next(layer for layer in data["layers"] if layer["layer"] == "examples")
        legacy.pop("reference_examples")
        legacy["suggestion"] = "可补充一个例子，或者说明不要示例。"
        # This was already invalid before the reference-layer migration.
        for model in (self.previous.LayerAnalysis, LayerAnalysis):
            with self.subTest(model=model), self.assertRaises(ValueError):
                model.model_validate(data)
        report = {"status": "unreviewed", "layers": data["layers"], "layer_decisions": [
            {"layer": "examples", "choice": "model", "value": legacy["suggestion"]}]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "early-report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            output = io.StringIO()
            self.assertTrue(launcher.show_report(path, output, details=True))
        self.assertIn("参考材料", output.getvalue())
        self.assertIn(legacy["suggestion"], output.getvalue())


if __name__ == "__main__":
    unittest.main()
