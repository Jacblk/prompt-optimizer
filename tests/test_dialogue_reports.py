import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from optimizer_config import ConfigurationError
from optimizer_engine import OptimizationResult
from optimizer_handoff import WindowLimits
import optimizer_reports as reports
from test_optimizer import configs


def result(status="unreviewed", text="已确认提示词", revision=1):
    return OptimizationResult(status, text, status == "ready", metadata={
        "request_count": 2, "known_total_tokens": 17,
        "dialogue": {"session_id": "test-session", "revision": revision,
                     "fingerprint": str(revision), "original_request": "需求",
                     "confirmed_request": "需求", "pending_questions": []}})


class DialogueSaveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.last = self.root / "last_optimized_prompt.md"
        self.last.write_text("之前的成功提示词", encoding="utf-8")

    def test_current_success_saves_report_and_prompt(self):
        outcome = reports.save_dialogue_result(result(), root=self.root, current=lambda r: True)
        self.assertTrue(outcome.report_saved)
        self.assertTrue(outcome.prompt_saved)
        self.assertFalse(outcome.errors)
        self.assertEqual(self.last.read_text(encoding="utf-8"), "已确认提示词")
        self.assertEqual(reports.load_report(outcome.report_path)["metadata"]["dialogue"]["revision"], 1)

    def test_every_non_success_checkpoint_protects_last_prompt(self):
        for status in ("needs_clarification", "needs_review", "cancelled", "failed",
                       "budget_exceeded", "context_exceeded", "configuration_error"):
            with self.subTest(status=status):
                outcome = reports.save_dialogue_result(result(status), root=self.root, current=lambda r: True)
                self.assertTrue(outcome.report_saved)
                self.assertFalse(outcome.prompt_saved)
                self.assertEqual(self.last.read_text(encoding="utf-8"), "之前的成功提示词")

    def test_stale_or_missing_version_gate_cannot_publish(self):
        for gate in (None, lambda r: False):
            outcome = reports.save_dialogue_result(result("ready"), root=self.root, current=gate)
            self.assertTrue(outcome.report_saved)
            self.assertFalse(outcome.prompt_saved)
        gate = Mock(side_effect=[True, False])
        outcome = reports.save_dialogue_result(result("ready"), root=self.root, current=gate)
        self.assertFalse(outcome.prompt_saved)
        self.assertEqual(gate.call_count, 2)
        self.assertEqual(self.last.read_text(encoding="utf-8"), "之前的成功提示词")

    def test_report_and_prompt_path_collisions_stop_all_writes(self):
        for path in (self.last, self.root / ".env", self.root / ".env.backup"):
            with self.subTest(path=path.name), patch.object(reports, "atomic_write") as writer:
                outcome = reports.save_dialogue_result(result("cancelled"), root=self.root, report_path=path)
                self.assertFalse(outcome.report_saved)
                self.assertTrue(outcome.errors)
                writer.assert_not_called()
        output = self.root / "same.md"
        outcome = reports.save_dialogue_result(result(), root=self.root, report_path=output,
                                               output_path=output, current=lambda r: True)
        self.assertTrue(outcome.errors)
        self.assertFalse(output.exists())

    def test_protected_sources_include_result_material_metadata(self):
        source = self.root / "source.md"
        source.write_text("原材料", encoding="utf-8")
        value = result()
        value.metadata["reference_files"] = {"files": [{"source": str(source)}]}
        outcome = reports.save_dialogue_result(value, root=self.root, output_path=source, current=lambda r: True)
        self.assertTrue(outcome.errors)
        self.assertEqual(source.read_text(encoding="utf-8"), "原材料")

    def test_hard_link_alias_of_source_is_protected(self):
        source = self.root / "source.txt"
        source.write_text("原材料", encoding="utf-8")
        alias = self.root / "alias.md"
        os.link(source, alias)
        outcome = reports.save_dialogue_result(result(), root=self.root, output_path=alias,
                                               protected_paths=[source], current=lambda r: True)
        self.assertTrue(outcome.errors)
        self.assertEqual(alias.read_text(encoding="utf-8"), "原材料")

    def test_report_failure_does_not_lose_current_prompt(self):
        writer = reports.atomic_write
        def fail_report(path, text):
            if path.suffix == ".json":
                raise OSError("simulated")
            return writer(path, text)
        with patch.object(reports, "atomic_write", side_effect=fail_report):
            outcome = reports.save_dialogue_result(result(), root=self.root, current=lambda r: True)
        self.assertFalse(outcome.report_saved)
        self.assertTrue(outcome.prompt_saved)
        self.assertTrue(outcome.errors)

    def test_prompt_failure_preserves_previous_success_and_keeps_report(self):
        writer = reports.atomic_write
        def fail_prompt(path, text):
            if path == self.last:
                raise OSError("simulated")
            return writer(path, text)
        with patch.object(reports, "atomic_write", side_effect=fail_prompt):
            outcome = reports.save_dialogue_result(result(), root=self.root, current=lambda r: True)
        self.assertTrue(outcome.report_saved)
        self.assertFalse(outcome.prompt_saved)
        self.assertTrue(outcome.errors)
        self.assertEqual(self.last.read_text(encoding="utf-8"), "之前的成功提示词")

    def test_draft_export_is_labelled_and_cannot_replace_last_success(self):
        draft = reports.save_draft("尚未确认", self.root / "draft.md")
        self.assertEqual(draft.read_text(encoding="utf-8"), "# 待确认草稿\n\n尚未确认")
        with self.assertRaises(ConfigurationError):
            reports.save_draft("尚未确认", self.last)
        self.assertEqual(self.last.read_text(encoding="utf-8"), "之前的成功提示词")


class ReportReadTests(unittest.TestCase):
    def test_old_report_and_invalid_report_are_read_without_config_or_models(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "old.json"
            legacy = {"status": "ready", "optimized_prompt": "旧结果", "metadata": {
                "dialogue": {"total_rounds": 3, "allowed_rounds": 3,
                             "question_rounds": [{"round_number": 3, "round_limit": 3}]}}}
            path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8-sig")
            with patch("optimizer_config.read_environment", side_effect=AssertionError("must stay offline")):
                self.assertEqual(reports.load_report(path)["optimized_prompt"], "旧结果")
                self.assertEqual(reports.load_report(path)["metadata"]["dialogue"]["allowed_rounds"], 3)
                for text in ('{"status":"ready","status":"failed"}', "[]", "broken", '{"status":NaN}',
                             '{"status":"ready","metadata":[]}', '{"status":"ready","candidates":["bad"]}',
                             '{"status":"ready","metadata":{"dialogue":[]}}'):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ConfigurationError):
                        reports.load_report(path)

    def test_env_report_rejected_before_read(self):
        with patch.object(Path, "open", side_effect=AssertionError("must not read configuration")):
            with self.assertRaises(ConfigurationError):
                reports.load_report(Path(".env"))


class WindowModalSaveTests(unittest.TestCase):
    def test_all_quality_roles_bound_publicly_after_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "context_windows.json"
            data = reports.configure_dialogue_windows(configs(),
                                                     {"a": "128000", "b": 128000, "judge": 128000}, path=path)
            WindowLimits.from_config(data, configs())
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("api_key", text)
            self.assertNotIn("base_url", text)
            self.assertEqual(set(data["roles"]), {"a", "b", "judge"})

    def test_cancel_partial_invalid_and_small_window_never_change_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "context_windows.json"
            original = '{"version":1,"roles":{}}\n'
            path.write_text(original, encoding="utf-8")
            for values in ({}, {"a": 128000}, {"a": "0", "b": 128000, "judge": 128000},
                           {"a": True, "b": 128000, "judge": 128000},
                           {"a": "not a size", "b": 128000, "judge": 128000},
                           {"a": 5, "b": 128000, "judge": 128000}):
                with self.subTest(values=values):
                    with self.assertRaises(ConfigurationError):
                        reports.configure_dialogue_windows(configs(), values, path=path)
                    self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_window_config_cannot_replace_success_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "last_optimized_prompt.md"
            path.write_text("之前的成功", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                reports.configure_dialogue_windows(configs(), {"a": 128000, "b": 128000, "judge": 128000}, path=path)
            self.assertEqual(path.read_text(encoding="utf-8"), "之前的成功")
