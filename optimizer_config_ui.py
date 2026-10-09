"""One Textual form for the standalone configuration app and optimizer settings."""
from __future__ import annotations

import asyncio

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static, TabbedContent, TabPane, TextArea

from optimizer_config import ConfigurationError, ModelConfig, handoff_model_config
from optimizer_settings import ConfigurationStore, SPECS, format_value


def field_id(key):
    return "cfg-" + key.replace(".", "-").replace("_", "-")


class ConfigurationScreen(ModalScreen[dict | None]):
    BINDINGS = [Binding("escape", "cancel", "取消")]
    DEFAULT_CSS = """
    ConfigurationScreen { align: center middle; background: $background 75%; }
    #configuration-dialog { width: 96%; max-width: 120; height: 94%; border: thick $primary;
        background: $surface; padding: 0 1; }
    #configuration-tabs { height: 1fr; min-height: 3; }
    .configuration-fields { height: 1fr; padding: 0 1; }
    .configuration-fields Label { height: auto; margin-top: 1; }
    .configuration-fields Input { height: 3; }
    .configuration-fields TextArea { height: 5; }
    .configuration-help { height: auto; color: $text-muted; }
    #configuration-info, #configuration-error { height: auto; }
    #configuration-error { color: $error; }
    #configuration-buttons { height: 3; }
    #configuration-buttons Button { width: 1fr; min-width: 8; margin-right: 1; }
    """

    def __init__(self, store):
        super().__init__()
        self.store = store
        self.baseline = store.form_values()
        self.initial = {key: format_value(value) for key, value in self.baseline.items()}
        self.public_values = store.public()
        self._saving = False

    def compose(self) -> ComposeResult:
        with Vertical(id="configuration-dialog"):
            yield Label("配置模型与参数 · 新会话生效")
            yield Static("密钥留空保留；交接输出在常用页，留空自动。保存与核验不调用模型。" +
                         " ".join(self.store.recovery_notes), classes="configuration-help")
            with TabbedContent(id="configuration-tabs"):
                for advanced, title in ((False, "常用"), (True, "高级")):
                    with TabPane(title, id="configuration-advanced" if advanced else "configuration-common"):
                        with VerticalScroll(classes="configuration-fields"):
                            for key, spec in SPECS.items():
                                if spec.advanced != advanced:
                                    continue
                                role = key.split(".")[1].upper() if key.startswith("models.") else ""
                                yield Label((f"模型 {role} · " if role else "") + spec.label)
                                if spec.kind == "json":
                                    yield TextArea(self.initial[key], id=field_id(key))
                                else:
                                    yield Input(self.initial[key], password=spec.secret, id=field_id(key),
                                                placeholder="留空保留已有密钥" if spec.secret else "留空" if spec.optional else "")
                                help_text = spec.help
                                if spec.secret:
                                    configured = self.public_values["models"][key.split(".")[1]]["api_key_configured"]
                                    help_text += " 当前：" + ("已配置。" if configured else "尚未配置。")
                                else:
                                    help_text += " 默认：" + (format_value(spec.default) if spec.kind != "json" else "{}") + "。"
                                    help_text += " 来源：" + self.public_values["sources"].get(key, "默认值") + "。"
                                if spec.choices:
                                    help_text += " 选项：" + " / ".join(map(str, spec.choices))
                                if spec.kind == "bool":
                                    help_text += " 选项：true / false"
                                if help_text:
                                    yield Static(help_text, classes="configuration-help")
            yield Static("", id="configuration-info")
            yield Static("", id="configuration-error")
            with Horizontal(id="configuration-buttons"):
                yield Button("保存", id="configuration-save", variant="primary")
                yield Button("核验", id="configuration-check")
                yield Button("默认值", id="configuration-defaults")
                yield Button("取消", id="configuration-cancel")

    def on_mount(self):
        self.refresh_info()

    def value(self, key):
        widget = self.query_one("#" + field_id(key))
        return widget.text if isinstance(widget, TextArea) else widget.value

    def patch(self):
        return {key: self.value(key) for key in SPECS if self.value(key) != self.initial[key]}

    def refresh_info(self):
        patch = self.patch()
        previews = []
        for role in ("a", "b", "c"):
            try:
                prefix = "models." + role + "."
                get = lambda key: SPECS[prefix + key].parse(self.value(prefix + key))
                tokens = get("max_tokens")
                if role == "b":
                    previews.append(f"B：提示词生成 {tokens}（普通／交接共用）")
                    continue
                extra = get("extra_body")
                effort = get("reasoning_effort")
                if effort is not None:
                    extra["reasoning_effort"] = effort
                else:
                    extra.pop("reasoning_effort", None)
                config = ModelConfig("judge" if role == "c" else role, get("name"),
                    "http://127.0.0.1:9/v1", "preview", max_tokens=tokens,
                    extra_body=extra, handoff_max_tokens=get("handoff_max_tokens"),
                    handoff_reasoning_effort=get("handoff_reasoning_effort"))
                history = handoff_model_config(config)
                usage = "提示词生成" if role == "a" else "候选评审"
                history_usage = "历史整理" if role == "a" else "历史核验"
                previews.append(f"{role.upper()}：{usage} {config.max_tokens} / {history_usage} {history.max_tokens}" +
                                ("（自动）" if config.handoff_max_tokens is None else ""))
            except ConfigurationError:
                previews.append(role.upper() + "：参数待修正")
        overridden = sum(source == "进程环境覆盖" for source in self.public_values["sources"].values())
        message = "\n".join(previews) + f"\n待保存 {len(patch)} 项"
        if overridden:
            message += f"；{overridden} 项被进程环境覆盖，保存不会改变该进程环境。"
        self.query_one("#configuration-info", Static).update(message)

    @on(Input.Changed)
    @on(TextArea.Changed)
    def field_changed(self):
        if self.is_mounted:
            self.refresh_info()

    def action_cancel(self):
        if not self._saving:
            self.dismiss(None)

    @on(Button.Pressed)
    async def pressed(self, event):
        event.stop()
        identifier = event.button.id
        if self._saving:
            return
        if identifier == "configuration-cancel":
            self.action_cancel()
        elif identifier == "configuration-defaults":
            for key, spec in SPECS.items():
                if spec.secret or key.endswith((".name", ".base_url", ".context_window")):
                    continue
                widget = self.query_one("#" + field_id(key))
                value = format_value(spec.default)
                if isinstance(widget, TextArea):
                    widget.load_text(value)
                else:
                    widget.value = value
        elif identifier in {"configuration-save", "configuration-check"}:
            self._saving = True
            for button in self.query("#configuration-buttons Button"):
                button.disabled = True
            try:
                result = await asyncio.to_thread(self.store.apply, self.patch(),
                    allow_secrets=True, dry_run=identifier == "configuration-check")
                if identifier == "configuration-save":
                    self.dismiss(result)
                else:
                    self.query_one("#configuration-error", Static).update(
                        "变更核验通过。" + ("；".join(result["warnings"]) if result["warnings"] else "配置完整有效。"))
            except (ConfigurationError, OSError, UnicodeError):
                import sys
                error = sys.exc_info()[1]
                self.query_one("#configuration-error", Static).update(
                    str(error) if isinstance(error, ConfigurationError) else "配置无法保存，请检查文件权限。")
            finally:
                self._saving = False
                for button in self.query("#configuration-buttons Button"):
                    button.disabled = False


class ConfigurationApp(App):
    TITLE = "提示词优化器配置"

    def __init__(self, *, root, store_factory=ConfigurationStore):
        super().__init__()
        self.root, self.store_factory = root, store_factory

    async def on_mount(self):
        try:
            store = await asyncio.to_thread(self.store_factory, self.root, recover=True)
            await self.push_screen(ConfigurationScreen(store), self.exit)
        except (ConfigurationError, OSError, UnicodeError) as error:
            self.exit(str(error) if isinstance(error, ConfigurationError) else "配置无法读取。")
