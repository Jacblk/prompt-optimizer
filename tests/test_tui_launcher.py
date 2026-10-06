import builtins
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import launcher
import optimize


class TuiLauncherTests(unittest.TestCase):
    def test_launches_tui_directly_once_without_menu_or_input(self):
        output = io.StringIO()
        run_tui = Mock(return_value=0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = launcher.main([], stdout=output, run_tui=run_tui, root=root)
            run_tui.assert_called_once_with(root=root)
        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue(), "")

    def test_compatibility_filename_opens_the_same_tui(self):
        run_tui = Mock(return_value=None)
        self.assertIs(optimize.main, launcher.main)
        self.assertEqual(optimize.main([], run_tui=run_tui), 0)
        run_tui.assert_called_once_with(root=launcher.ROOT)

    def test_tui_exit_code_is_propagated(self):
        self.assertEqual(launcher.main([], run_tui=Mock(return_value=7)), 7)

    def test_missing_textual_gives_single_install_instruction_and_fails(self):
        errors = io.StringIO()
        run_tui = Mock(side_effect=ModuleNotFoundError("textual", name="textual"))
        self.assertEqual(launcher.main([], stderr=errors, run_tui=run_tui), 1)
        self.assertIn("requirements.txt", errors.getvalue())
        self.assertIn("只需为当前环境安装一次", errors.getvalue())
        run_tui.assert_called_once()

    def test_help_and_old_flags_do_not_import_ui_or_load_configuration(self):
        importer = builtins.__import__

        def deny_product(name, *args, **kwargs):
            if name == "tui" or name.startswith(("textual", "optimizer_config", "legacy_")):
                raise AssertionError("must stop before UI or configuration import")
            return importer(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=deny_product):
            output, errors = io.StringIO(), io.StringIO()
            self.assertEqual(launcher.main(["--help"], stdout=output), 0)
            self.assertIn("TUI", output.getvalue())
            self.assertIn("历史有内容时自动交接", output.getvalue())
            self.assertIn("A/B 生成、C 独立评审", output.getvalue())
            self.assertNotIn("设置模式", output.getvalue())
            self.assertEqual(launcher.main(["--request", "旧需求"], stderr=errors), 2)
            self.assertIn("不再接受旧命令行参数", errors.getvalue())
            self.assertNotIn("设置模式", errors.getvalue())

    def test_startup_error_does_not_display_service_details(self):
        for error in (RuntimeError("private-service-detail"), ImportError("private-service-detail")):
            with self.subTest(error=type(error).__name__):
                output = io.StringIO()
                self.assertEqual(launcher.main([], stderr=output, run_tui=Mock(side_effect=error)), 1)
                self.assertNotIn("private-service-detail", output.getvalue())

    def test_keyboard_interrupt_has_cancel_exit_code(self):
        self.assertEqual(launcher.main([], run_tui=Mock(side_effect=KeyboardInterrupt)), 130)

    def test_import_does_not_load_ui_or_retired_cli(self):
        script = (
            "import sys; import launcher; import optimize; "
            "assert optimize.main is launcher.main; "
            "assert 'textual' not in sys.modules; "
            "assert 'optimizer_config' not in sys.modules; "
            "assert 'legacy_optimize' not in sys.modules; "
            "assert 'legacy_launcher' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=launcher.ROOT,
                                capture_output=True, timeout=15, check=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))

    @unittest.skipUnless(os.name == "nt", "Windows launcher")
    def test_cmd_launches_tui_from_another_directory_with_utf8_output(self):
        with tempfile.TemporaryDirectory(prefix="optimizer launch ") as temp:
            # Exercise the real CMD/launcher without a UI or model configuration.
            fake_ui = (
                "import json, sys, types\n"
                "ui = types.ModuleType('tui')\n"
                "def start_fake_ui(*, root):\n"
                "    print('对话界面 ' + json.dumps(str(root), ensure_ascii=False))\n"
                "    return 0\n"
                "ui.main = start_fake_ui\n"
                "sys.modules['tui'] = ui\n"
            )
            (Path(temp) / "sitecustomize.py").write_text(fake_ui, encoding="utf-8")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = temp
            environment["PYTHONIOENCODING"] = "ascii"
            result = subprocess.run(
                [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c",
                 str(launcher.ROOT / "启动提示词优化器.cmd")],
                cwd=temp, env=environment, capture_output=True, input=b"", timeout=15, check=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        self.assertIn("对话界面 " + json.dumps(str(launcher.ROOT), ensure_ascii=False),
                      result.stdout.decode("utf-8"))
        self.assertEqual(result.stderr, b"")
