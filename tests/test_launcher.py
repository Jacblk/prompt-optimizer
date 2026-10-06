import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import legacy_launcher as launcher
import legacy_optimize as optimize
from optimizer_engine import Optimizer
from optimizer_models import ModelCallError
from test_optimizer import ORIGINAL, Scripts, configs, draft, review


class LauncherTests(unittest.TestCase):
    def invoke(self, text, root, scripts):
        output = io.StringIO()

        def run(args, **streams):
            return optimize.main(args, **streams, optimizer_factory=lambda c, o, **extra:
                                 Optimizer(c, o, factory=scripts.factory, **extra))

        with patch("legacy_optimize.ROOT", root), patch("legacy_optimize.read_environment", return_value={}), \
                patch("legacy_optimize.load_models", return_value=configs()):
            code = launcher.main(stdin=io.StringIO(text), stdout=output,
                                 run_optimizer=run, root=root)
        self.assertEqual(code, 0)
        return output.getvalue()

    def test_original_selection_is_explained_and_report_can_be_reopened_without_calls(self):
        scripts = Scripts(judge=[lambda p: review(p, "keep_original")])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = self.invoke(f"1\n2\n{ORIGINAL}\nEND\nr\n0\n", root, scripts)
            paths = list((root / "reports").glob("*.json"))
            self.assertEqual(len(paths), 1)
            data = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(data["optimized_prompt"], ORIGINAL + "\n")
            self.assertEqual((root / "last_optimized_prompt.md").read_text(encoding="utf-8"), ORIGINAL + "\n")
        self.assertEqual(len(scripts.calls), 4)
        self.assertIn("采用：原文", output)
        self.assertIn("原因：已检查。", output)
        self.assertIn("第 1 次评审", output)
        self.assertIn("甲结果", output)
        self.assertIn("乙结果", output)

    def test_historical_menu_preserves_pasted_text_with_the_current_pipeline(self):
        scripts = Scripts()
        request = "--不改文件\n{保留中文与花括号}"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text = f"\n\n{request}\nEND\n1\n2\n第二份\nEND\n1\n3\n第三份\nEND\n0\n"
            output = self.invoke(text, root, scripts)
            reports = [json.loads(p.read_text(encoding="utf-8"))
                       for p in sorted((root / "reports").glob("*.json"))]
        self.assertEqual(len(reports), 3)
        self.assertEqual([r["metadata"]["request_count"] for r in reports], [4, 4, 4])
        self.assertTrue(all("strict_review" not in r["metadata"]["limits"] for r in reports))
        self.assertEqual(scripts.calls[0][2]["original_request"], request + "\n")
        self.assertEqual(len(scripts.calls), 12)
        self.assertIn("评审通过", output)

    def test_failed_run_keeps_previous_report_and_explains_that_it_is_older(self):
        scripts = Scripts(a=[draft("甲第一次成功"), ModelCallError("离线模拟失败")],
                          b=[draft("乙结果"), ModelCallError("离线模拟失败")])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = self.invoke("1\n1\n第一次\nEND\n1\n1\n第二次\nEND\nr\n0\n", root, scripts)
            self.assertEqual(len(list((root / "reports").glob("*.json"))), 1)
            self.assertEqual((root / "last_optimized_prompt.md").read_text(encoding="utf-8"), "甲第一次成功")
        self.assertEqual(len(scripts.calls), 7)
        self.assertIn("两个生成器均未产出有效候选", output)
        self.assertIn("本次没有保存新的评审报告", output)
        self.assertIn("报告文件：", output)

    def test_pending_result_is_saved_but_does_not_replace_previous_success(self):
        pending = draft("待确认草稿", status="needs_clarification", clarification_questions=["检查哪些问题？"])
        scripts = Scripts(a=[pending], b=[pending], judge=[lambda p:
                          review(p, "needs_clarification") | {"clarification_questions": ["检查哪些问题？"]}])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = root / "last_optimized_prompt.md"
            saved.write_text("之前的结果", encoding="utf-8")
            output = self.invoke("1\n1\n检查\nEND\nr\n0\n", root, scripts)
            self.assertEqual(saved.read_text(encoding="utf-8"), "之前的结果")
            data = json.loads(launcher.newest_report(root).read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "needs_clarification")
        self.assertEqual(len(scripts.calls), 4)
        self.assertIn("待确认：检查哪些问题？", output)
        self.assertIn("待确认草稿", output)
        self.assertNotIn("采用：", output)

    def test_selected_changes_are_shown_as_generator_claims_without_extra_calls(self):
        scripts = Scripts(a=[draft("甲明确的任务", change_summary=["整理受众与语气。"] )])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = self.invoke("1\n1\n原始需求\nEND\n0\n", root, scripts)
            report = json.loads(launcher.newest_report(root).read_text(encoding="utf-8"))
        self.assertEqual(len(scripts.calls), 4)
        self.assertEqual(next(candidate["origin"] for candidate in report["candidates"]
                              if candidate["id"] == report["selected_id"]), "a")
        self.assertTrue(report["reviewed"])
        self.assertIn("改写说明（生成器自述）：整理受众与语气。", output)
        self.assertIn("评审通过", output)

    def test_quality_summary_uses_only_the_selected_candidates_changes(self):
        scripts = Scripts(a=[draft("甲结果", change_summary=["甲的改动"])],
                          b=[draft("乙结果", change_summary=["乙的改动"])])
        with tempfile.TemporaryDirectory() as temp:
            output = self.invoke("1\n2\n原始需求\nEND\n0\n", Path(temp), scripts)
        self.assertEqual(len(scripts.calls), 4)
        self.assertIn("改写说明（生成器自述）：甲的改动", output)
        self.assertNotIn("乙的改动", output)

    def test_empty_cancelled_and_report_only_actions_never_load_config_or_run_model(self):
        for text in ("r\nwrong\n1\n1\n\nEND\n0\n", "1\n1\n尚未输入结束标记\n", "0\n"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as temp, \
                    patch("legacy_optimize.read_environment", side_effect=AssertionError("config access")):
                runner = Mock(side_effect=AssertionError("model access"))
                self.assertEqual(launcher.main(stdin=io.StringIO(text), stdout=io.StringIO(),
                                              run_optimizer=runner, root=Path(temp)), 0)
                runner.assert_not_called()

    def test_bad_report_does_not_close_menu_or_trigger_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reports").mkdir()
            (root / "reports" / "broken.json").write_text("{broken", encoding="utf-8")
            output = io.StringIO()
            runner = Mock(side_effect=AssertionError("model access"))
            self.assertEqual(launcher.main(stdin=io.StringIO("r\n0\n"), stdout=output,
                                          run_optimizer=runner, root=root), 0)
        runner.assert_not_called()
        self.assertIn("报告无法读取或格式不完整", output.getvalue())

if __name__ == "__main__":
    unittest.main()
