"""Shared configuration, transaction, runtime and CLI contracts; synthetic secrets only."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cli
from optimizer_config import ConfigurationError, load_models, read_environment
from optimizer_engine import Optimizer, RunOptions
from optimizer_reports import validate_dialogue_paths
from optimizer_settings import (
    ConfigurationStore, SPECS, default_settings, history_options, read_settings,
    reference_options, run_options,
)
from test_optimizer import Scripts, ORIGINAL, review, engine
from test_dialogue import DialogueScripts, sufficient
from test_handoff import HistoryScripts, runner
from test_dialogue_handoff import models


def seed_models(root, configs=None):
    configs = configs or models()
    prefixes = {"a": "GENERATOR_A", "b": "GENERATOR_B", "judge": "JUDGE"}
    lines = ["# keep this comment", "UNRELATED='literal ${opaque}'"]
    for role, config in configs.items():
        prefix = prefixes[role]
        for suffix, value in (("NAME", config.name), ("BASE_URL", config.base_url),
                              ("API_KEY", config.api_key), ("MAX_TOKENS", config.max_tokens),
                              ("EXTRA_BODY", json.dumps(config.extra_body))):
            lines.append(f"{prefix}_{suffix}='{value}'")
    (Path(root) / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return configs


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="配置 中文 ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.configs = seed_models(self.root)

    def store(self, **kwargs):
        return ConfigurationStore(self.root, **kwargs)

    def invoke(self, arguments, **kwargs):
        output, errors = io.StringIO(), io.StringIO()
        with patch("optimizer_models.LangChainChatModel.complete", side_effect=AssertionError("offline config")):
            code = cli.main(arguments, root=self.root, stdout=output, stderr=errors,
                            stdin=kwargs.pop("stdin", io.StringIO()), **kwargs)
        return code, output.getvalue(), errors.getvalue()

    def test_runtime_default_and_example_are_equal_and_never_read_credentials(self):
        with patch("dotenv.dotenv_values", side_effect=AssertionError("no env read")):
            self.assertEqual(read_settings(self.root), default_settings())
            sample = json.loads((Path(__file__).resolve().parents[1] / "optimizer_settings.example.json").read_text())
            self.assertEqual(sample, default_settings())
            self.assertEqual(run_options(sample).retries, 1)
            self.assertEqual(reference_options(sample).chunk_overlap, 160)
            self.assertEqual(history_options(sample).chunk_bytes, 24000)

    def test_bulk_patch_preserves_comments_unrelated_values_and_credentials(self):
        result = self.store().apply({"run.network_max_attempts": "9", "models.a.max_tokens": "384000",
                                     "models.a.context_window": "1000000",
                                     "history.chunk_bytes": "12345", "reference.chunk_overlap": "120"})
        self.assertIn("run.network_max_attempts", result["changed"])
        text = (self.root / ".env").read_text()
        self.assertIn("# keep this comment", text)
        self.assertIn("UNRELATED='literal ${opaque}'", text)
        values = read_environment(self.root / ".env")
        self.assertEqual(values["GENERATOR_A_API_KEY"], self.configs["a"].api_key)
        self.assertEqual(load_models(values)["a"].max_tokens, 384000)
        settings = read_settings(self.root)
        self.assertEqual(run_options(settings).retries, 8)
        self.assertEqual(history_options(settings).chunk_bytes, 12345)
        self.assertEqual(reference_options(settings).chunk_overlap, 120)
        self.assertEqual(self.store().public()["models"]["a"]["context_window"], 1000000)

    def test_invalid_batch_changes_nothing(self):
        original = (self.root / ".env").read_bytes()
        with self.assertRaises(ConfigurationError):
            self.store().apply({"models.a.max_tokens": "200000", "reference.chunk_overlap": "1600"})
        self.assertEqual((self.root / ".env").read_bytes(), original)
        self.assertFalse((self.root / "optimizer_settings.json").exists())

    def test_secrets_are_hidden_and_only_explicit_hidden_input_can_replace(self):
        sentinel = "fake-key-with-'quote'\\backslash-${literal}"
        self.store().apply({"models.a.api_key": sentinel}, allow_secrets=True)
        self.assertEqual(read_environment(self.root / ".env")["GENERATOR_A_API_KEY"], sentinel)
        self.assertEqual(self.store().form_values()["models.a.api_key"], "")
        self.assertNotIn(sentinel, json.dumps(self.store().public()))
        with self.assertRaises(ConfigurationError) as caught:
            self.store().apply({"models.a.api_key": "never-echo-argument"})
        self.assertNotIn("never-echo", str(caught.exception))
        self.store().apply({"models.a.api_key": ""}, allow_secrets=True)
        self.assertEqual(read_environment(self.root / ".env")["GENERATOR_A_API_KEY"], sentinel)
        self.store().apply({"models.a.api_key": "   "}, allow_secrets=True)
        self.assertEqual(read_environment(self.root / ".env")["GENERATOR_A_API_KEY"], sentinel)

    def test_single_quote_and_backslash_round_trip_and_duplicate_managed_keys(self):
        with (self.root / ".env").open("a", encoding="utf-8") as stream:
            stream.write("GENERATOR_A_NAME='duplicate'\n")
        self.store().apply({"models.a.name": "model 'literal'\\path"})
        values = read_environment(self.root / ".env")
        self.assertEqual(values["GENERATOR_A_NAME"], "model 'literal'\\path")
        self.assertEqual((self.root / ".env").read_text().count("GENERATOR_A_NAME="), 1)

    def test_managed_assignment_preserves_inline_comments_and_export_prefix(self):
        with (self.root / ".env").open("a", encoding="utf-8") as stream:
            stream.write("export GENERATOR_A_MAX_TOKENS = 2048  # retain numeric comment\n")
            stream.write('GENERATOR_B_NAME="old # within value" # retain quoted comment\n')
        self.store().apply({"models.a.max_tokens": 32000, "models.b.name": "updated"})
        saved = (self.root / ".env").read_text()
        self.assertIn("export GENERATOR_A_MAX_TOKENS = '32000'  # retain numeric comment", saved)
        self.assertIn("GENERATOR_B_NAME='updated' # retain quoted comment", saved)

    def test_model_change_invalidates_only_its_window_and_requires_reconfirmation(self):
        self.store().apply({f"models.{role}.context_window": 128000 for role in ("a", "b", "c")})
        self.assertTrue(self.store().check()["valid"])
        self.store().apply({"models.a.name": "replacement"})
        windows = json.loads((self.root / "context_windows.json").read_text())["roles"]
        self.assertEqual(set(windows), {"b", "judge"})
        self.assertFalse(self.store().check()["valid"])
        self.store().apply({"models.a.context_window": 128000})
        self.assertTrue(self.store().check()["valid"])

    def test_auto_history_limit_and_explicit_override_are_visible(self):
        self.store().apply({"models.a.name": "deepseek-flash", "models.a.reasoning_effort": "high",
                            "models.a.context_window": 128000})
        view = self.store().public()["models"]["a"]
        self.assertEqual(view["effective_handoff_max_tokens"], 65536)
        self.store().apply({"models.a.handoff_max_tokens": "200000", "models.a.context_window": 500000})
        self.assertEqual(self.store().public()["models"]["a"]["effective_handoff_max_tokens"], 200000)

    def test_unknown_version_duplicates_nonfinite_and_resource_limits_fail(self):
        for text in ('{"version":2}', '{"version":1,"version":1}', '{"version":1,"run":{"token_budget":NaN}}',
                     '{"version":1,"history":{"max_file_bytes":5242881}}',
                     '{"version":1,"reference":{"max_pdf_pages":201}}'):
            with self.subTest(text=text):
                (self.root / "optimizer_settings.json").write_text(text)
                with self.assertRaises(ConfigurationError):
                    read_settings(self.root)

    def test_failed_multifile_write_rolls_back_exact_original_bytes(self):
        store = self.store()
        original = (self.root / ".env").read_bytes()
        from optimizer_io import atomic_write_bytes
        calls = []
        def fail_second(path, data):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("private-error-not-returned")
            atomic_write_bytes(path, data)
        with patch("optimizer_settings.atomic_write_bytes", side_effect=fail_second):
            with self.assertRaises(ConfigurationError) as caught:
                store.apply({"models.a.max_tokens": 16000, "run.network_max_attempts": 3})
        self.assertIn("回滚", str(caught.exception))
        self.assertNotIn("private-error", str(caught.exception))
        self.assertEqual((self.root / ".env").read_bytes(), original)
        self.assertFalse((self.root / "optimizer_settings.json").exists())

    def test_concurrent_change_and_source_or_hardlink_destinations_are_rejected(self):
        store = self.store()
        with (self.root / ".env").open("a") as stream:
            stream.write("# another writer\n")
        with self.assertRaises(ConfigurationError):
            store.apply({"run.network_max_attempts": 3})
        (self.root / "app.py").write_text("source")
        with self.assertRaises(ConfigurationError):
            self.store(env_path=self.root / "app.py")
        alias = self.root / "alias.env"
        os.link(self.root / ".env", alias)
        with self.assertRaises(ConfigurationError):
            self.store(env_path=alias)

    def test_window_template_cannot_be_used_as_a_configuration_or_output_target(self):
        template = self.root / "context_windows.example.json"
        sample = Path(__file__).resolve().parents[1] / template.name
        template.write_bytes(sample.read_bytes())
        original = template.read_bytes()
        alias = self.root / "template-alias.json"
        for target in (template, alias):
            if target == alias:
                os.link(template, alias)
            for field in ("env_path", "context_path"):
                with self.subTest(target=target.name, field=field), patch.object(
                        Path, "read_bytes", side_effect=AssertionError("read before target validation")):
                    with self.assertRaises(ConfigurationError):
                        self.store(**{field: target})
            for field in ("output_path", "report_path"):
                with self.subTest(target=target.name, field=field), self.assertRaises(ConfigurationError):
                    validate_dialogue_paths(root=self.root, **{field: target})
        self.assertEqual(template.read_bytes(), original)

    def test_first_configuration_creates_local_windows_and_preserves_the_empty_template(self):
        template = self.root / "context_windows.example.json"
        template.write_bytes((Path(__file__).resolve().parents[1] / template.name).read_bytes())
        original = template.read_bytes()
        self.assertEqual(json.loads(original), {"version": 1, "roles": {}})
        windows = self.root / "context_windows.json"
        self.assertFalse(windows.exists())
        initial = self.store().check()
        self.assertTrue(initial["generation_ready"])
        self.assertFalse(initial["handoff_ready"])
        code, text, errors = self.invoke(["--configure-contexts", "128000", "128000", "128000"])
        self.assertEqual(code, 0, errors)
        self.assertTrue(self.store().check()["handoff_ready"])
        self.assertEqual(set(json.loads(windows.read_text())["roles"]), {"a", "b", "judge"})
        self.assertEqual(template.read_bytes(), original)

    def test_dry_run_does_not_write_and_process_override_is_reported(self):
        with patch.dict(os.environ, {"GENERATOR_A_MAX_TOKENS": "32000"}):
            store = self.store()
            self.assertEqual(store.public()["sources"]["models.a.max_tokens"], "进程环境覆盖")
            self.assertEqual(store.public()["models"]["a"]["max_tokens"], 32000)
            store.apply({"models.a.max_tokens": 64000}, dry_run=True)
        self.assertEqual(load_models(read_environment(self.root / ".env"))["a"].max_tokens, self.configs["a"].max_tokens)

    def test_cli_view_check_set_and_legacy_contexts_share_the_store(self):
        code, text, errors = self.invoke(["--config-set", "run.network_max_attempts", "8",
                                        "--config-set", "models.a.max_tokens", "200000", "--json"])
        self.assertEqual(code, 0, errors)
        self.assertEqual(json.loads(text)["status"], "configured")
        code, text, errors = self.invoke(["--configure-contexts", "500000", "128000", "128000"])
        self.assertEqual(code, 0, errors)
        self.assertTrue(self.store().check()["valid"])
        code, text, errors = self.invoke(["--config-show", "--json"])
        self.assertEqual(code, 0, errors)
        view = json.loads(text)
        self.assertEqual(view["run"]["network_max_attempts"], 8)
        self.assertEqual(view["models"]["a"]["max_tokens"], 200000)
        self.assertNotIn(self.configs["a"].api_key, text)
        code, text, errors = self.invoke(["--config-check", "--json"])
        self.assertEqual(code, 0, errors)
        self.assertTrue(json.loads(text)["valid"])

    def test_cli_rejects_secret_arguments_duplicate_keys_and_mixed_generation(self):
        original = (self.root / ".env").read_bytes()
        for args in (["--config-set", "models.a.api_key", "do-not-echo"],
                     ["--config-set", "run.network_max_attempts", "3", "--config-set", "run.network_max_attempts", "4"],
                     ["--config-show", "demand"], ["--configure", "--retries", "1"],
                     ["--config-set", "run.token_budget", "NaN"]):
            with self.subTest(args=args):
                code, text, errors = self.invoke(args)
                self.assertEqual(code, 2, errors)
                self.assertNotIn("do-not-echo", text + errors)
        self.assertEqual((self.root / ".env").read_bytes(), original)

    def test_runtime_defaults_cli_overrides_and_actual_engine_options(self):
        self.store().apply({"run.network_max_attempts": 8, "run.prompt_max_repairs": 3,
                            "run.schema_max_attempts": 4, "run.token_budget": 100000,
                            "run.handoff_max_attempts": 5})
        made = []
        scripts = DialogueScripts()
        def factory(configs, options, **kwargs):
            made.append(options)
            return Optimizer(configs, options, factory=scripts.factory, **kwargs)
        code, text, errors = self.invoke(["demand", "--non-interactive", "--retries", "0",
                                        "--prompt-max-repairs", "0", "--no-token-budget"],
                                       optimizer_factory=factory)
        self.assertEqual(code, 0, errors)
        self.assertEqual(made[0].retries, 0)
        self.assertEqual(made[0].schema_max_attempts, 4)
        self.assertEqual(made[0].handoff_max_attempts, 5)
        self.assertEqual(made[0].prompt_max_repairs, 0)
        self.assertIsNone(made[0].token_budget)

    def test_same_window_value_explicitly_confirms_replaced_model(self):
        self.store().apply({"models.a.context_window": 128000})
        self.store().apply({"models.a.name": "new-model", "models.a.context_window": 128000})
        view = self.store().public()["models"]["a"]
        self.assertEqual(view["name"], "new-model")
        self.assertEqual(view["context_window"], 128000)

    def test_explicit_recovery_is_previewed_and_does_not_write_until_saved(self):
        path = self.root / "optimizer_settings.json"
        path.write_text("invalid original parameter file", encoding="utf-8")
        original = path.read_bytes()
        with self.assertRaises(ConfigurationError):
            self.store()
        recovering = self.store(recover=True)
        self.assertTrue(recovering.recovery_notes)
        recovering.apply({}, dry_run=True)
        self.assertEqual(path.read_bytes(), original)
        recovering.apply({})
        self.assertEqual(read_settings(self.root), default_settings())

    def test_invalid_model_fields_can_be_corrected_without_echoing_private_url(self):
        with (self.root / ".env").open("a", encoding="utf-8") as stream:
            stream.write("GENERATOR_A_MAX_TOKENS='invalid'\n")
            stream.write("GENERATOR_A_TEMPERATURE='null'\n")
            stream.write("GENERATOR_A_JSON_MODE=' TRUE '\n")
            stream.write('GENERATOR_A_EXTRA_BODY=\'{"reasoning_effort":{"hidden":"private-extra-value"}}\'\n')
            stream.write("GENERATOR_A_BASE_URL='https://private-user:hidden-password@example.invalid/v1?token=hidden-token'\n")
        store = self.store()
        self.assertEqual(store.form_values()["models.a.max_tokens"], "invalid")
        self.assertEqual(store.form_values()["models.a.temperature"], "null")
        self.assertIs(store.form_values()["models.a.json_mode"], True)
        view = json.dumps(store.public())
        self.assertNotIn("hidden-password", view)
        self.assertNotIn("hidden-token", view)
        self.assertNotIn("private-extra-value", view)
        store.apply({"models.a.max_tokens": 32000, "models.a.base_url": "https://example.invalid/v1",
                     "models.a.temperature": None, "models.a.reasoning_effort": None})
        self.assertEqual(self.store().public()["models"]["a"]["max_tokens"], 32000)
        self.assertTrue(self.store().check()["generation_ready"])

    def test_partial_runtime_file_sources_and_all_window_issues_are_reported(self):
        (self.root / "optimizer_settings.json").write_text('{"version":1,"run":{"network_max_attempts":3}}')
        store = self.store()
        self.assertEqual(store.public()["sources"]["run.network_max_attempts"], "参数文件")
        self.assertEqual(store.public()["sources"]["run.schema_max_attempts"], "默认值")
        check = store.check()
        self.assertTrue(check["generation_ready"])
        self.assertFalse(check["handoff_ready"])
        self.assertEqual(len(check["errors"]), 3)

    def test_interactive_wizard_uses_hidden_keys_and_returns_only_public_summary(self):
        class Tty(io.StringIO):
            def isatty(self):
                return True
        answers = ["n"]
        for key, spec in SPECS.items():
            if not spec.advanced and not spec.secret:
                answers.append("4" if key == "run.network_max_attempts" else "")
        answers.append("y")
        with patch("getpass.getpass", side_effect=["wizard-hidden-value", "", ""]) as hidden:
            code, output, errors = self.invoke(["--configure", "--json"], stdin=Tty("\n".join(answers) + "\n"))
        self.assertEqual(code, 0, errors)
        self.assertEqual(hidden.call_count, 3)
        self.assertNotIn("wizard-hidden-value", output + errors)
        self.assertNotIn(self.configs["a"].api_key, output + errors)
        self.assertEqual(read_environment(self.root / ".env")["GENERATOR_A_API_KEY"], "wizard-hidden-value")
        self.assertEqual(read_settings(self.root)["run"]["network_max_attempts"], 4)

    def test_wizard_cancel_or_eof_preserves_files_and_json_status(self):
        class Tty(io.StringIO):
            def isatty(self):
                return True
        original = (self.root / ".env").read_bytes()
        for text in ("n\n3\n/cancel\n", ""):
            code, output, errors = self.invoke(["--configure", "--json"], stdin=Tty(text))
            self.assertEqual(code, 130, errors)
            self.assertEqual(json.loads(output)["status"], "cancelled")
            self.assertEqual((self.root / ".env").read_bytes(), original)
            self.assertFalse((self.root / "optimizer_settings.json").exists())

    def test_parameter_file_cannot_be_used_as_generation_input_or_output(self):
        self.store().apply({"run.network_max_attempts": 3})
        path = self.root / "optimizer_settings.json"
        original = path.read_bytes()
        for arguments in (["--input", str(path)], ["demand", "--output", str(path)],
                          ["demand", "--report", str(path)], ["--show-report", str(path)]):
            code, _, errors = self.invoke(arguments)
            self.assertEqual(code, 2, errors)
            self.assertEqual(path.read_bytes(), original)
        from optimizer_documents import check_reference_path
        with self.assertRaises(ConfigurationError):
            check_reference_path(path)


class ConfiguredAttemptsTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_multiple_repairs_are_fully_reviewed_and_bounded(self):
        scripts = Scripts(judge=[lambda p: review(p, "repair"), lambda p: review(p, "repair"),
                                 lambda p: review(p, "select")])
        optimizer = engine(scripts, prompt_max_repairs=2)
        result = await optimizer.run(ORIGINAL)
        self.assertEqual(result.status, "ready", result.reason)
        self.assertEqual([c["purpose"] for c in result.metadata["calls"]].count("repair"), 2)
        self.assertEqual(len(optimizer.review_history), 3)

    async def test_history_three_attempts_use_actual_limit_and_stop_without_snapshot(self):
        from optimizer_models import OutputTruncatedError
        error = OutputTruncatedError("truncated", usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})
        scripts = HistoryScripts(handoff_extract=[error, error, error])
        result = await runner(scripts, handoff_max_attempts=3).run(ORIGINAL)
        self.assertEqual(result.status, "handoff_output_truncated")
        self.assertEqual(result.metadata["request_count"], 3)
        self.assertEqual(result.metadata["known_total_tokens"], 9)
        self.assertIsNone(result.metadata["handoff"]["snapshot"])

    async def test_schema_can_exceed_two_attempts_then_succeed(self):
        from optimizer_models import ModelReply
        scripts = DialogueScripts(clarification=[ModelReply("invalid", 1, 1, 2, "fake"),
                                                ModelReply("invalid", 1, 1, 2, "fake"), sufficient()])
        optimizer = Optimizer(models(), RunOptions(schema_max_attempts=3), factory=scripts.factory)
        from optimizer_dialogue import DialogueRequest
        result = await optimizer.run_dialogue(DialogueRequest(ORIGINAL))
        self.assertEqual(result.status, "ready", result.reason)
        self.assertEqual(sum(c["purpose"].startswith("dialogue_clarification") for c in result.metadata["calls"]), 3)
        await optimizer.aclose()

    async def test_prompt_repair_quota_is_shared_by_later_revisions_and_zero_disables(self):
        from test_dialogue import make_controller, review as dialogue_review
        scripts = DialogueScripts(judge=[lambda p: dialogue_review(p, "repair"),
            lambda p: dialogue_review(p, "repair"), lambda p: dialogue_review(p, "select"),
            lambda p: dialogue_review(p, "repair")])
        controller, _ = make_controller(scripts, prompt_max_repairs=2)
        self.addAsyncCleanup(controller.close)
        first = await controller.submit(ORIGINAL)
        self.assertEqual(first.status, "ready", first.reason)
        second = await controller.submit("继续保持只读，补充异常检查。")
        self.assertEqual(second.status, "needs_review", second.reason)
        self.assertIn("次数上限", second.reason)
        self.assertEqual(second.metadata["dialogue"]["prompt_repairs_used"], 2)
        self.assertEqual(sum("repair" in payload for _, _, payload, _ in scripts.calls), 2)
        disabled, disabled_scripts = make_controller(DialogueScripts(
            judge=[lambda p: dialogue_review(p, "repair")]), prompt_max_repairs=0)
        self.addAsyncCleanup(disabled.close)
        result = await disabled.submit(ORIGINAL)
        self.assertEqual(result.status, "needs_review")
        self.assertFalse(any("repair" in payload for _, _, payload, _ in disabled_scripts.calls))


class StandaloneConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_standalone_app_reads_shared_values_saves_and_exits_at_narrow_size(self):
        from optimizer_config_ui import ConfigurationApp, ConfigurationScreen, field_id
        from textual.widgets import Input
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_models(root)
            app = ConfigurationApp(root=root)
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                self.assertIsInstance(app.screen, ConfigurationScreen)
                key = app.screen.query_one("#" + field_id("models.a.api_key"), Input)
                self.assertTrue(key.password)
                self.assertEqual(key.value, "")
                app.screen.query_one("#" + field_id("run.network_max_attempts"), Input).value = "9"
                await pilot.click("#configuration-save")
                await pilot.pause()
            self.assertEqual(app.return_value["status"], "configured")
            self.assertEqual(ConfigurationStore(root).public()["run"]["network_max_attempts"], 9)

    async def test_common_handoff_inputs_are_editable_saved_and_all_roles_are_previewed(self):
        from optimizer_config_ui import ConfigurationApp, field_id
        from textual.widgets import Input, Static, TabbedContent
        for size in ((80, 24), (120, 40)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                seed_models(root)
                ConfigurationStore(root).apply({"models.a.name": "deepseek-flash",
                    "models.a.max_tokens": 8192, "models.b.max_tokens": 5120,
                    "models.c.max_tokens": 4096,
                    **{f"models.{role}.context_window": 128000 for role in ("a", "b", "c")}})
                app = ConfigurationApp(root=root)
                async with app.run_test(size=size) as pilot:
                    await pilot.pause()
                    self.assertEqual(app.screen.query_one("#configuration-tabs", TabbedContent).active,
                                     "configuration-common")
                    info = str(app.screen.query_one("#configuration-info", Static).content)
                    self.assertIn("A：提示词生成 8192 / 历史整理 65536（自动）", info)
                    self.assertIn("B：提示词生成 5120（普通／交接共用）", info)
                    self.assertIn("C：候选评审 4096 / 历史核验 4096（自动）", info)
                    for role, value in (("a", "98304"), ("c", "12000")):
                        key = f"models.{role}.handoff_max_tokens"
                        field = app.screen.query_one("#configuration-common #" + field_id(key), Input)
                        self.assertFalse(field.disabled)
                        field.value = ""
                        field.focus()
                        await pilot.pause()
                        await pilot.press(*value)
                        self.assertEqual(field.value, value)
                    await pilot.pause()
                    info = str(app.screen.query_one("#configuration-info", Static).content)
                    self.assertIn("历史整理 98304", info)
                    self.assertIn("历史核验 12000", info)
                    self.assertNotIn("（自动）", info)
                    self.assertIn("B：提示词生成 5120", info)
                    buttons = app.screen.query_one("#configuration-save")
                    self.assertGreaterEqual(buttons.region.y, app.screen.query_one("#configuration-info").region.bottom)
                    self.assertTrue(await pilot.click("#configuration-save"))
                    await pilot.pause()
                view = ConfigurationStore(root).public()["models"]
                self.assertEqual(view["a"]["effective_handoff_max_tokens"], 98304)
                self.assertEqual(view["b"]["max_tokens"], 5120)
                self.assertEqual(view["c"]["effective_handoff_max_tokens"], 12000)
