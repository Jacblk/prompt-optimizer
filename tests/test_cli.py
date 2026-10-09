"""CLI contracts against the real engine and scripted models; no real .env/API."""
import asyncio
import builtins
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cli
from optimizer_config import ConfigurationError
from optimizer_engine import Optimizer
from optimizer_reports import configure_dialogue_windows
from test_dialogue import ORIGINAL, DialogueScripts, ask, question, review, sufficient
from test_dialogue_handoff import audited, extracted, models


class Terminal(io.StringIO):
    def isatty(self):
        return True


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="optimizer CLI 中文 ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.last = self.root / "last_optimized_prompt.md"
        self.last.write_text("之前的成功提示词", encoding="utf-8")
        self.configs = models()

    def invoke(self, args, *, scripts=None, stdin=None, stdout=None, loader=None, factory=None):
        scripts = scripts or DialogueScripts()
        output = stdout if stdout is not None else io.StringIO()
        errors = io.StringIO()
        loader = loader or Mock(return_value=self.configs)
        factory = factory or (lambda configs, options, **kwargs: Optimizer(
            configs, options, factory=scripts.factory, rng=random.Random(42), **kwargs))
        with patch("optimizer_config.read_environment", side_effect=AssertionError("no real .env")):
            code = cli.main(args, root=self.root, stdin=stdin if stdin is not None else io.StringIO(),
                stdout=output, stderr=errors, config_loader=loader, optimizer_factory=factory)
        return code, output.getvalue(), errors.getvalue(), scripts, loader

    def test_help_is_stdlib_only_and_does_not_read_input_or_config(self):
        importer = builtins.__import__
        def deny(name, *args, **kwargs):
            if name.startswith(("optimizer_config", "optimizer_engine", "textual", "tui")):
                raise AssertionError("help must stay lazy")
            return importer(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=deny):
            output = io.StringIO()
            self.assertEqual(cli.main(["--help"], stdout=output, stdin=Mock()), 0)
        self.assertIn("--non-interactive", output.getvalue())
        self.assertIn("--configure-contexts", output.getvalue())

    def test_invalid_inputs_stop_before_config_or_model(self):
        invalid = [[], [""], ["TEXT", "--input", "missing.md"], ["--token-budget", "0"],
                   ["TEXT", "--retries", "-1"], ["--unknown"], ["--preview-references"],
                   ["TEXT", "--context-config", "windows.json"], ["TEXT", "--reference-mode", "relevant"],
                   ["TEXT", "--interactive"], ["--input", str(self.root / "missing.md")],
                   ["--show-report", "a.json", "TEXT"], ["--configure-contexts", "1", "2", "3", "TEXT"],
                   ["TEXT", "--history-file", "h.md", "--history-text", "history"],
                   ["TEXT", "--no-env", "--env-file", "settings.txt"]]
        for args in invalid:
            with self.subTest(args=args):
                code, output, errors, scripts, loader = self.invoke(args)
                self.assertEqual(code, 2, errors)
                self.assertEqual(output, "")
                loader.assert_not_called()
                self.assertEqual(scripts.calls, [])
        self.assertEqual(self.last.read_text(encoding="utf-8"), "之前的成功提示词")

    def test_text_sources_keep_unicode_and_stdout_contains_only_prompt(self):
        source = self.root / "需求.md"
        source.write_text(ORIGINAL + "\n中文 😀", encoding="utf-8-sig")
        variants = [([ORIGINAL], io.StringIO(), ORIGINAL),
                    (["-p", ORIGINAL], io.StringIO(), ORIGINAL),
                    (["--request", ORIGINAL], io.StringIO(), ORIGINAL),
                    (["--input", str(source)], io.StringIO(), ORIGINAL + "\n中文 😀"),
                    ([], io.StringIO("\ufeff" + ORIGINAL), ORIGINAL),
                    (["--input", "-"], io.StringIO(ORIGINAL), ORIGINAL)]
        for args, stream, expected in variants:
            with self.subTest(args=args):
                code, output, errors, scripts, loader = self.invoke(args, stdin=stream)
                self.assertEqual(code, 0, errors)
                self.assertEqual(output, expected + "\n")
                self.assertEqual(self.last.read_text(encoding="utf-8"), expected)
                self.assertEqual(len(scripts.calls), 4)
                self.assertCountEqual(scripts.closed, ["a", "b", "judge"])
                loader.assert_called_once_with(self.root / ".env")

    def test_json_has_usage_and_files_and_respects_paths(self):
        report, output = self.root / "out" / "report.json", self.root / "out" / "result.md"
        code, body, errors, _, _ = self.invoke([ORIGINAL, "--json", "--quiet", "-o", str(output), "--report", str(report)])
        data = json.loads(body)
        self.assertEqual(code, 0, errors)
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["metadata"]["request_count"], 4)
        self.assertEqual(data["files"], {"report": str(report), "prompt": str(output)})
        self.assertEqual(data["save_errors"], [])
        self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["optimized_prompt"], ORIGINAL)
        self.assertEqual(output.read_text(encoding="utf-8"), ORIGINAL)
        self.assertEqual(errors, "")
        self.assertEqual(self.last.read_text(encoding="utf-8"), "之前的成功提示词")

    def test_interactive_multiline_request_and_answers_use_existing_dialogue(self):
        scripts = DialogueScripts(clarification=[ask(question()), sufficient()])
        stream = Terminal(ORIGINAL + "\n第二行\nEND\n/multi\n只检查异常处理\n输出清单\nEND\n")
        code, body, errors, _, _ = self.invoke([], scripts=scripts, stdin=stream, stdout=Terminal())
        self.assertEqual(code, 0, errors)
        self.assertIn(ORIGINAL + "\n第二行", body)
        self.assertIn("只检查异常处理\n输出清单", body)
        reports = [json.loads(p.read_text(encoding="utf-8")) for p in (self.root / "reports").glob("*.json")]
        self.assertEqual({r["status"] for r in reports}, {"ready", "needs_clarification"})

    def test_explicit_option_adoption_and_multiple_rounds(self):
        options = [{"id": "read", "label": "仅检查错误"}, {"id": "write", "label": "修改文件"}]
        scripts = DialogueScripts(clarification=[ask(question(options=options)), ask(question()), sufficient()])
        code, body, errors, _, _ = self.invoke([ORIGINAL, "--interactive", "--json"],
            scripts=scripts, stdin=Terminal("/9\n/1\n输出表格\n"))
        self.assertEqual(code, 0, errors)
        data = json.loads(body)["metadata"]["dialogue"]
        self.assertEqual(data["total_rounds"], 2)
        self.assertEqual(data["updates"][0]["option_id"], "read")
        self.assertNotIn("修改文件", data["confirmed_request"])
        self.assertIn("没有这个选项", errors)

    def test_partial_answers_leave_remaining_question_unconfirmed(self):
        scripts = DialogueScripts(clarification=[ask(question("one"), question("two", "验收方式？")),
            lambda payload: ask(payload["pending_questions"][0])])
        code, body, errors, _, _ = self.invoke([ORIGINAL, "--interactive", "--json"],
            scripts=scripts, stdin=Terminal("仅查错误\n/skip\n/pause\n"))
        self.assertEqual(code, 3, errors)
        data = json.loads(body)["metadata"]["dialogue"]
        self.assertEqual(data["updates"][0]["raw_text"], "仅查错误")
        self.assertEqual(len(data["pending_questions"]), 1)

    def test_noninteractive_pending_questions_never_read_answers_or_overwrite_success(self):
        class NoRead(Terminal):
            def readline(self, *args):
                raise AssertionError("must not wait for answers")
        for options in (["--non-interactive"], []):
            with self.subTest(options=options):
                scripts = DialogueScripts(clarification=[ask(question())])
                code, body, errors, _, _ = self.invoke([ORIGINAL, "--json", *options], scripts=scripts, stdin=NoRead())
                self.assertEqual(code, 3, errors)
                data = json.loads(body)
                self.assertEqual(data["status"], "needs_clarification")
                self.assertIsNone(data["files"]["prompt"])
                self.assertEqual(len(scripts.calls), 1)
                self.assertIn("需要检查哪些问题", errors)
        self.assertEqual(self.last.read_text(encoding="utf-8"), "之前的成功提示词")

    def test_empty_eof_and_pause_do_not_adopt_options(self):
        for answer in ("\n", "", "/pause\n"):
            scripts = DialogueScripts(clarification=[ask(question(options=[{"id": "write", "label": "修改文件"}]))])
            code, body, errors, _, _ = self.invoke([ORIGINAL, "--interactive", "--json"],
                scripts=scripts, stdin=Terminal(answer))
            self.assertEqual(code, 3, errors)
            self.assertEqual(json.loads(body)["metadata"]["dialogue"]["confirmed_request"], ORIGINAL)
            self.assertEqual(len(scripts.calls), 1)

    def test_cancel_command_and_keyboard_interrupt_save_cancelled_report(self):
        class Interrupted(Terminal):
            def readline(self, *args):
                raise KeyboardInterrupt
        for stream in (Terminal("/cancel\n"), Interrupted()):
            scripts = DialogueScripts(clarification=[ask(question())])
            code, body, errors, _, _ = self.invoke([ORIGINAL, "--interactive", "--json"], scripts=scripts, stdin=stream)
            self.assertEqual(code, 130, errors)
            self.assertEqual(json.loads(body)["status"], "cancelled")
            self.assertIsNotNone(json.loads(body)["files"]["report"])
        self.assertEqual(self.last.read_text(encoding="utf-8"), "之前的成功提示词")

    def test_cancel_during_model_and_budget_failure_keep_previous_result(self):
        for scripts, options, expected, status in (
                (DialogueScripts(clarification=[asyncio.CancelledError()]), [], 130, "cancelled"),
                (DialogueScripts(), ["--token-budget", "1"], 1, "budget_exceeded")):
            code, body, errors, _, _ = self.invoke([ORIGINAL, "--json", *options], scripts=scripts)
            self.assertEqual(code, expected, errors)
            self.assertEqual(json.loads(body)["status"], status)
            self.assertEqual(self.last.read_text(encoding="utf-8"), "之前的成功提示词")

    def test_needs_review_returns_four_without_publishing_draft(self):
        scripts = DialogueScripts(judge=[lambda payload: review(payload, "repair")])
        code, body, errors, _, _ = self.invoke([ORIGINAL, "--no-repair"], scripts=scripts)
        self.assertEqual(code, 4, errors)
        self.assertEqual(body, "")
        self.assertEqual(self.last.read_text(encoding="utf-8"), "之前的成功提示词")

    def test_preflight_rejects_protected_paths_and_aliases_before_config(self):
        source = self.root / "input.md"
        source.write_text(ORIGINAL, encoding="utf-8")
        alias = self.root / "alias.md"
        os.link(source, alias)
        for target in (source, alias, self.root / "cli.py", self.root / "prompt-optimizer.cmd",
                       self.root / "CLI_WORKFLOW.md", self.root / ".env", self.root):
            if target.suffix == ".py":
                target.write_text("# source", encoding="utf-8")
            with self.subTest(target=target):
                code, _, errors, scripts, loader = self.invoke(["-i", str(source), "-o", str(target)])
                self.assertEqual(code, 2, errors)
                loader.assert_not_called()
                self.assertFalse(scripts.calls)
        code, _, _, _, loader = self.invoke([ORIGINAL, "--report", str(self.last)])
        self.assertEqual(code, 2)
        loader.assert_not_called()
        code, _, _, _, loader = self.invoke([ORIGINAL, "-o", str(alias), "--report", str(alias)])
        self.assertEqual(code, 2)
        loader.assert_not_called()
        self.assertEqual(source.read_text(encoding="utf-8"), ORIGINAL)

    def test_configuration_cannot_be_loaded_as_input(self):
        for option in ("--input", "--reference-file", "--history-file", "--show-report"):
            args = [option, str(self.root / ".env")]
            if option in {"--reference-file", "--history-file"}:
                args.insert(0, ORIGINAL)
            code, _, _, _, loader = self.invoke(args)
            self.assertEqual(code, 2)
            loader.assert_not_called()
        custom = self.root / "secrets.txt"
        code, _, _, _, loader = self.invoke(["--input", str(custom), "--env-file", str(custom)])
        self.assertEqual(code, 2)
        loader.assert_not_called()
        code, _, _, _, loader = self.invoke(["--configure-contexts", "128000", "128000", "128000",
                                            "--context-config", str(custom), "--env-file", str(custom)])
        self.assertEqual(code, 2)
        loader.assert_not_called()

    def test_reference_preview_is_offline_and_includes_provenance(self):
        source = self.root / "reference.md"
        source.write_text("# 范例\n只读检查 src，输出列表。", encoding="utf-8")
        code, body, errors, scripts, loader = self.invoke(["--preview-references", "--reference-file", str(source), "--json"])
        self.assertEqual(code, 0, errors)
        data = json.loads(body)
        self.assertTrue(data["files"][0]["covers_all_text"])
        self.assertEqual(len(data["files"][0]["file_sha256"]), 64)
        self.assertIn("只读检查 src", data["blocks"][0])
        self.assertFalse(scripts.calls)
        loader.assert_not_called()
        self.assertFalse((self.root / "reports").exists())

    def test_reference_scope_reaches_shared_engine(self):
        source = self.root / "reference.md"
        source.write_text("# 范例\n只读检查 src，输出列表。", encoding="utf-8")
        code, body, errors, scripts, _ = self.invoke([ORIGINAL, "--reference-file", str(source),
            "--reference-purpose", "结构参考", "--reference-usage", "仅借鉴布局", "--json"])
        self.assertEqual(code, 0, errors)
        data = json.loads(body)
        self.assertIn("仅借鉴布局", data["optimized_prompt"])
        self.assertEqual(data["metadata"]["reference_files"]["files"][0]["purpose"], "结构参考")
        self.assertIn("只读检查 src", json.dumps(scripts.calls[0][2], ensure_ascii=False))

    def test_show_report_is_offline_does_not_publish_or_restore(self):
        path = self.root / "old.json"
        path.write_text(json.dumps({"status": "ready", "optimized_prompt": "旧报告结果"}), encoding="utf-8")
        code, body, errors, scripts, loader = self.invoke(["--show-report", str(path), "--json"])
        self.assertEqual(code, 0, errors)
        self.assertEqual(json.loads(body)["optimized_prompt"], "旧报告结果")
        loader.assert_not_called()
        self.assertFalse(scripts.calls)
        self.assertEqual(self.last.read_text(encoding="utf-8"), "之前的成功提示词")

    def test_config_sources_and_window_configuration_do_not_call_models(self):
        for options, expected in ((["--no-env"], None), (["--env-file", str(self.root / "custom.env")], self.root / "custom.env")):
            code, _, errors, scripts, loader = self.invoke(["--configure-contexts", "128000", "128000", "128000", *options])
            self.assertEqual(code, 0, errors)
            loader.assert_called_once_with(expected)
            self.assertFalse(scripts.calls)
            saved = (self.root / "context_windows.json").read_text(encoding="utf-8")
            self.assertNotIn("api_key", saved)
            self.assertNotIn("base_url", saved)

    def test_history_preparation_and_window_validation_precede_model_calls(self):
        for args in ([ORIGINAL, "--history-file", str(self.root / "missing.md")],
                     [ORIGINAL, "--history-text", ""],
                     [ORIGINAL, "--history-text", "用户：只读。"]):
            code, _, errors, scripts, _ = self.invoke(args)
            self.assertEqual(code, 2, errors)
            self.assertFalse(scripts.calls)

    def test_history_runs_real_shared_handoff_pipeline(self):
        configure_dialogue_windows(self.configs, {role: 256000 for role in self.configs}, path=self.root / "context_windows.json")
        scripts = DialogueScripts(a=[extracted], judge=[audited])
        history = self.root / "history.md"
        history.write_text("用户：只读，不改文件。", encoding="utf-8")
        code, body, errors, _, _ = self.invoke([ORIGINAL, "--history-file", str(history), "--json"], scripts=scripts)
        self.assertEqual(code, 0, errors)
        data = json.loads(body)
        self.assertEqual(data["metadata"]["workflow"], "handoff")
        self.assertIn("handoff", data["metadata"])
        self.assertEqual(data["metadata"]["request_count"], 6)

    def test_errors_are_safe_and_json_runtime_errors_are_machine_readable(self):
        for error, expected in ((ConfigurationError("缺少配置项：GENERATOR_A_NAME"), 2),
                                (RuntimeError("private-provider-detail"), 1)):
            code, body, errors, _, _ = self.invoke([ORIGINAL, "--json"], loader=Mock(side_effect=error))
            self.assertEqual(code, expected)
            self.assertEqual(json.loads(body)["status"], "error")
            self.assertNotIn("private-provider-detail", body + errors)

    def test_report_failure_returns_one_but_retains_successful_stdout(self):
        from optimizer_io import atomic_write
        def fail_report(path, text):
            if path.suffix == ".json":
                raise OSError("private-path-detail")
            atomic_write(path, text)
        with patch("optimizer_reports.atomic_write", side_effect=fail_report):
            code, body, errors, _, _ = self.invoke([ORIGINAL])
        self.assertEqual(code, 1)
        self.assertEqual(body, ORIGINAL + "\n")
        self.assertEqual(self.last.read_text(encoding="utf-8"), ORIGINAL)
        self.assertNotIn("private-path-detail", errors)


class CliProcessTests(unittest.TestCase):
    def test_sigint_during_model_request_saves_cancelled_report(self):
        script = """
import asyncio
from pathlib import Path
import signal
import sys
sys.path[:0] = [sys.argv[1], str(Path(sys.argv[1]) / 'tests')]
import cli
from optimizer_engine import Optimizer
from test_dialogue import DialogueScripts
from test_dialogue_handoff import models
async def slow(payload):
    asyncio.get_running_loop().call_later(0.01, signal.raise_signal, signal.SIGINT)
    await asyncio.sleep(60)
scripts = DialogueScripts(clarification=[slow])
cli.configure_stdio()
raise SystemExit(cli.main(['only inspect src', '--json', '--quiet'], root=sys.argv[2],
    config_loader=lambda path: models(),
    optimizer_factory=lambda c, o, **kw: Optimizer(c, o, factory=scripts.factory, **kw)))
"""
        with tempfile.TemporaryDirectory() as temp:
            last = Path(temp) / "last_optimized_prompt.md"
            last.write_text("previous success", encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", script, str(cli.ROOT), temp],
                cwd=temp, input=b"", capture_output=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 130, result.stderr.decode("utf-8", errors="replace"))
            data = json.loads(result.stdout.decode("utf-8"))
            self.assertEqual(data["status"], "cancelled")
            self.assertIsNone(data["files"]["prompt"])
            self.assertEqual(json.loads(Path(data["files"]["report"]).read_text(encoding="utf-8"))["status"], "cancelled")
            self.assertEqual(last.read_text(encoding="utf-8"), "previous success")

    def test_utf8_pipeline_through_real_process_preserves_json_and_exit_status(self):
        script = (
            "import sys; from pathlib import Path; "
            "sys.path[:0] = [sys.argv[1], str(Path(sys.argv[1]) / 'tests')]; "
            "import cli; from optimizer_engine import Optimizer; "
            "from test_dialogue import DialogueScripts; from test_dialogue_handoff import models; "
            "cli.configure_stdio(); scripts = DialogueScripts(); "
            "raise SystemExit(cli.main(['--json', '--quiet'], root=sys.argv[2], "
            "config_loader=lambda path: models(), "
            "optimizer_factory=lambda c, o, **kw: Optimizer(c, o, factory=scripts.factory, **kw)))"
        )
        with tempfile.TemporaryDirectory() as temp:
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "ascii"
            result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", script, str(cli.ROOT), temp],
                cwd=temp, env=environment, input=(ORIGINAL + "\n中文 😀").encode("utf-8"),
                capture_output=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
            data = json.loads(result.stdout.decode("utf-8"))
            self.assertEqual(data["optimized_prompt"], ORIGINAL + "\n中文 😀")
            self.assertEqual(Path(data["files"]["prompt"]).parent, Path(temp).resolve())
            self.assertEqual(result.stderr, b"")

    def test_import_and_help_work_without_site_packages_or_tui(self):
        script = ("import sys, cli; assert 'optimizer_config' not in sys.modules; "
                  "assert 'textual' not in sys.modules; raise SystemExit(cli.main(['--help']))")
        result = subprocess.run([sys.executable, "-B", "-S", "-X", "utf8", "-c", script], cwd=cli.ROOT,
                                capture_output=True, timeout=15, check=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        self.assertIn("提示词优化器 CLI", result.stdout.decode("utf-8"))

    @unittest.skipUnless(os.name == "nt", "Windows launcher")
    def test_cmd_from_other_directory_uses_project_python_and_forwards_arguments(self):
        with tempfile.TemporaryDirectory(prefix="cli foreign cwd ") as temp:
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "ascii"
            for arguments, expected in ((["--help"], 0), (["--unknown"], 2),
                                        (["--show-report", "missing report.json", "--json"], 2)):
                command = subprocess.list2cmdline([str(cli.ROOT / "prompt-optimizer.cmd"), *arguments])
                result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", command],
                    cwd=temp, env=environment, input=b"", capture_output=True, timeout=20, check=False)
                self.assertEqual(result.returncode, expected, result.stderr.decode("utf-8", errors="replace"))
                if expected == 0:
                    self.assertIn("提示词优化器 CLI", result.stdout.decode("utf-8"))
                    self.assertEqual(result.stderr, b"")
                elif "--json" in arguments:
                    self.assertEqual(json.loads(result.stdout.decode("utf-8"))["status"], "error")


if __name__ == "__main__":
    unittest.main()
