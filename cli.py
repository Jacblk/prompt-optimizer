"""Terminal and pipeline interface to the shared dialogue engine.

Import and --help are stdlib-only. Configuration is loaded only for an explicit
generation or context-window configuration action; previews remain offline.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import sys

from optimizer_io import configure_stdio

ROOT = Path(__file__).resolve().parent
EXIT_CODES = {"ready": 0, "unreviewed": 0, "needs_clarification": 3,
              "needs_review": 4, "cancelled": 130}


class UsageError(Exception):
    pass


class Parser(argparse.ArgumentParser):
    def __init__(self, *, stdout, stderr, **kwargs):
        super().__init__(**kwargs)
        self.stdout, self.stderr = stdout, stderr

    def _print_message(self, message, file=None):
        if message:
            target = self.stderr if file is sys.stderr else self.stdout
            target.write(message)

    def error(self, message):
        raise UsageError(message)


def positive_int(value):
    try:
        number = int(value)
        if number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("须为正整数")


def build_parser(*, stdout, stderr):
    parser = Parser(stdout=stdout, stderr=stderr, prog="prompt-optimizer",
        description="提示词优化器 CLI：对话澄清、A/B 生成、C 独立评审。",
        epilog="不传需求时：终端中输入多行并以 END 提交；管道中读取 UTF-8 标准输入。")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("text", nargs="?", metavar="TEXT", help="待优化的需求或提示词")
    source.add_argument("-p", "--prompt", "--request", help="待优化的需求或提示词")
    source.add_argument("-i", "--input", metavar="FILE", help="UTF-8 文本文件；- 表示标准输入")
    interactive = parser.add_mutually_exclusive_group()
    interactive.add_argument("--interactive", action="store_true", help="允许终端澄清（即使输出已重定向）")
    interactive.add_argument("--non-interactive", action="store_true", help="有未决问题时保存报告并退出，不等待输入")
    parser.add_argument("-o", "--output", type=Path, help="成功提示词保存位置；默认项目 last_optimized_prompt.md")
    parser.add_argument("--report", type=Path, help="JSON 报告位置；默认项目 reports/ 下的时间戳文件")
    parser.add_argument("--json", action="store_true", help="stdout 输出完整 JSON 结果和保存路径")
    parser.add_argument("-q", "--quiet", action="store_true", help="隐藏进度和保存提示；仍显示问题及错误")
    config = parser.add_mutually_exclusive_group()
    config.add_argument("--env-file", type=Path, help="模型配置文件；默认项目 .env")
    config.add_argument("--no-env", action="store_true", help="仅使用进程环境变量")
    parser.add_argument("--token-budget", type=positive_int, help="可选的会话 token 调度预算")
    parser.add_argument("--retries", type=int, choices=range(6), default=1, help="网络失败重试次数（默认 1）")
    parser.add_argument("--no-repair", action="store_true", help="关闭一次自动修复")
    material = parser.add_argument_group("材料与交接")
    material.add_argument("--reference-file", type=Path, action="append", default=[], help="本地参考文件，可重复")
    material.add_argument("--reference-mode", choices=("full", "relevant"), default="full", help="全文或关键词相关片段")
    material.add_argument("--reference-purpose", help="这些参考文件的用途")
    material.add_argument("--reference-usage", help="这些参考文件的遵循范围")
    material.add_argument("--reference-kind", choices=("auto", "implementation", "artifact", "execution"), default="auto")
    history = material.add_mutually_exclusive_group()
    history.add_argument("--history-file", type=Path, action="append", default=[], help="旧记录 TXT/Markdown，可按顺序重复")
    history.add_argument("--history-text", help="独立提供的旧记录文本；有历史时自动交接")
    material.add_argument("--context-config", type=Path, help="交接窗口配置；默认项目 context_windows.json")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preview-references", action="store_true", help="离线预览选中的参考材料，不读取模型配置")
    action.add_argument("--show-report", type=Path, help="离线回看 JSON 报告，不恢复会话")
    action.add_argument("--configure-contexts", type=positive_int, nargs=3, metavar=("A", "B", "C"),
                        help="填写并保存三个角色的上下文窗口，不调用模型")
    return parser


def is_terminal(stream):
    return bool(getattr(stream, "isatty", lambda: False)())


@contextmanager
def interruptible_input():
    # asyncio.run's SIGINT handler cancels a task but cannot unblock readline.
    # While no model is running, let Ctrl+C interrupt the terminal read directly.
    previous = None
    try:
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    except ValueError:  # An embedding caller may run outside the main thread.
        pass
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


def read_multiline(stdin):
    lines = []
    with interruptible_input():
        for line in stdin:
            if line.rstrip("\r\n") == "END":
                break
            lines.append(line)
    return "".join(lines)


def checked_text(text):
    text = text.lstrip("\ufeff").strip()
    if not text or "\x00" in text:
        raise UsageError("需求须为非空文本，不能包含 NUL 字符。")
    return text


def check_input_path(path, env_file):
    from optimizer_reports import _same_path
    path = Path(path).expanduser()
    resolved = path.resolve()
    if (any(item.name.casefold().startswith(".env") for item in (path, resolved))
            or env_file is not None and _same_path(resolved, env_file.expanduser().resolve())):
        raise UsageError("配置文件不能作为需求、历史、材料或报告读取。")
    return resolved


def read_request(args, stdin, stderr, env_file):
    supplied = args.text if args.text is not None else args.prompt
    if supplied is not None:
        return checked_text(supplied)
    if args.input and args.input != "-":
        path = check_input_path(args.input, env_file)
        try:
            return checked_text(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError):
            raise UsageError("需求文件无法读取；请使用 UTF-8 文本文件。") from None
    if args.preview_references and args.input is None:
        return ""
    if is_terminal(stdin):
        if args.non_interactive:
            raise UsageError("非交互模式须提供 TEXT、--input 文件或管道输入。")
        print("请输入需求，可换行；单独一行 END 提交，Ctrl+C 取消。", file=stderr, flush=True)
        return checked_text(read_multiline(stdin))
    try:
        return checked_text(stdin.read())
    except UnicodeError:
        raise UsageError("标准输入无法按 UTF-8 读取；请使用 UTF-8 管道或 --input 文件。") from None


def validate_arguments(args):
    has_source = any(value is not None for value in (args.text, args.prompt, args.input))
    has_history = bool(args.history_file) or args.history_text is not None
    if args.preview_references and not args.reference_file:
        raise UsageError("--preview-references 需要至少一个 --reference-file。")
    if args.preview_references and (has_history or args.output or args.report):
        raise UsageError("材料预览不接受历史或保存路径。")
    if (args.show_report or args.configure_contexts) and (
            has_source or has_history or args.reference_file or args.output or args.report):
        raise UsageError("回看报告、配置窗口须单独执行，不能同时提交需求、材料、历史或保存路径。")
    if args.context_config and not (has_history or args.configure_contexts):
        raise UsageError("--context-config 仅用于历史交接或 --configure-contexts。")
    if not args.reference_file and (args.reference_purpose or args.reference_usage
            or args.reference_mode != "full" or args.reference_kind != "auto"):
        raise UsageError("材料设置需要至少一个 --reference-file。")


def print_json(data, stream):
    print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), file=stream)


class TerminalSession:
    def __init__(self, args, *, stdin, stderr, root, protected, interactive):
        self.args, self.stdin, self.stderr = args, stdin, stderr
        self.root, self.protected, self.interactive = root, protected, interactive
        self.checkpoint_errors = []
        self.seen_calls = set()
        self.slow_calls = set()

    def event(self, event):
        if event.get("kind") == "questions" and event.get("result") is not None:
            outcome = self.save(event["result"])
            self.checkpoint_errors.extend(outcome.errors)
        if self.args.quiet:
            return
        for call in event.get("active_calls", []):
            identifier = call["request_id"]
            role = {"a": "A", "b": "B", "judge": "C"}.get(call["role"], call["role"])
            if identifier not in self.seen_calls:
                self.seen_calls.add(identifier)
                purpose = call["purpose"]
                phase = ("确认需求" if purpose.startswith("dialogue_clarification") else
                         "整理并核对历史" if purpose.startswith("handoff") else
                         {"generate": "生成提示词", "review": "独立评审", "repair": "修复提示词",
                          "repair_review": "复核修复结果"}.get(purpose, "处理需求"))
                print(f"[{role} #{identifier}] {phase}…", file=self.stderr, flush=True)
            if call.get("slow_waiting") and identifier not in self.slow_calls:
                self.slow_calls.add(identifier)
                print(f"[{role} #{identifier}] 等待较久，仍在等待响应；Ctrl+C 可取消。", file=self.stderr, flush=True)
            elif not call.get("slow_waiting"):
                self.slow_calls.discard(identifier)

    def save(self, result, *, current=None):
        from optimizer_reports import save_dialogue_result
        return save_dialogue_result(result, root=self.root, current=current,
            output_path=self.args.output, report_path=self.args.report, protected_paths=self.protected)

    def questions(self, batch):
        from optimizer_dialogue import DialogueAnswer, DialogueResponse
        print("\n需要确认：", file=self.stderr)
        answers = []
        for question in batch.questions:
            print(f"{question.id}. {question.text}\n  原因：{question.reason}", file=self.stderr)
            for index, option in enumerate(question.options, 1):
                print(f"  /{index}  {option.label}", file=self.stderr)
            if not self.interactive:
                continue
            print("回答后回车；/数字 选建议，/multi 多行，/skip 跳过，/pause 保存退出，/cancel 取消。\n"
                  "本轮回答一起提交；暂停或取消不提交本轮回答。",
                  file=self.stderr, flush=True)
            try:
                while True:
                    with interruptible_input():
                        raw = self.stdin.readline()
                    if not raw:
                        return DialogueResponse(answers=answers) if answers else DialogueResponse(action="pause")
                    text = raw.strip()
                    if text in {"/pause", "/cancel"}:
                        return DialogueResponse(action=text[1:])
                    if not text or text == "/skip":
                        break
                    if text == "/multi":
                        print("单独一行 END 提交：", file=self.stderr, flush=True)
                        answers.append(DialogueAnswer(question_id=question.id, text=read_multiline(self.stdin).strip()))
                        break
                    if text.startswith("/") and text[1:].isdigit():
                        index = int(text[1:]) - 1
                        if 0 <= index < len(question.options):
                            answers.append(DialogueAnswer(question_id=question.id, option_id=question.options[index].id))
                            break
                        print("没有这个选项，请重新输入。", file=self.stderr, flush=True)
                        continue
                    answers.append(DialogueAnswer(question_id=question.id, text=text))
                    break
            except KeyboardInterrupt:
                return DialogueResponse(action="cancel")
        return DialogueResponse(answers=answers) if self.interactive else DialogueResponse(action="pause")


async def run_session(args, *, text, configs, references, history, windows,
                      terminal, stdout, optimizer_factory):
    from optimizer_dialogue import DialogueController
    from optimizer_engine import Optimizer, RunOptions
    options = RunOptions(token_budget=args.token_budget, retries=args.retries, allow_repair=not args.no_repair)
    engine = (optimizer_factory or Optimizer)(configs, options, references=references,
                                               history=history, context_windows=windows)
    controller = DialogueController(engine, on_questions=terminal.questions, on_event=terminal.event)
    try:
        result = await controller.submit(text)
        # Save before closing the session: closing retires its publication gate.
        outcome = terminal.save(result, current=controller.is_current)
        errors = list(dict.fromkeys([*terminal.checkpoint_errors, *outcome.errors]))
        if args.json:
            print_json({**result.to_dict(), "files": {
                "report": str(outcome.report_path) if outcome.report_saved else None,
                "prompt": str(outcome.prompt_path) if outcome.prompt_saved else None},
                "save_errors": errors}, stdout)
        elif result.status in {"ready", "unreviewed"} and result.optimized_prompt:
            print(result.optimized_prompt, file=stdout)
        if result.status not in {"ready", "unreviewed"}:
            print(f"状态：{result.status}。{result.reason}", file=terminal.stderr)
        for warning in result.warnings:
            print(f"提示：{warning}", file=terminal.stderr)
        for error in errors:
            print(error, file=terminal.stderr)
        if not args.quiet:
            if outcome.report_saved:
                print(f"报告：{outcome.report_path}", file=terminal.stderr)
            if outcome.prompt_saved:
                print(f"提示词：{outcome.prompt_path}", file=terminal.stderr)
        code = EXIT_CODES.get(result.status, 1)
        return 1 if errors and code == 0 else code
    finally:
        await controller.close()


def execute(args, *, stdin, stdout, stderr, root, config_loader, optimizer_factory):
    from optimizer_config import load_models, read_environment
    from optimizer_documents import DEFAULT_PURPOSE, DEFAULT_USAGE, ReferenceFile, ReferenceOptions, prepare_references
    from optimizer_handoff import WindowLimits, prepare_history, read_window_config
    from optimizer_reports import configure_dialogue_windows, load_report, validate_dialogue_paths

    env_file = None if args.no_env else (args.env_file or root / ".env")
    context_path = args.context_config or root / "context_windows.json"
    if args.show_report:
        data = load_report(check_input_path(args.show_report, env_file))
        if args.json:
            print_json(data, stdout)
        else:
            if data.get("optimized_prompt"):
                print(data["optimized_prompt"], file=stdout)
            print(f"报告状态：{data['status']}（离线回看）", file=stderr)
            for question in data.get("questions", []):
                print(f"待确认：{question}", file=stderr)
        return 0
    loader = config_loader or (lambda path: load_models(read_environment(path)))
    if args.configure_contexts:
        check_input_path(context_path, env_file)
        configs = loader(env_file)
        configure_dialogue_windows(configs, dict(zip(("a", "b", "judge"), args.configure_contexts)), path=context_path)
        if args.json:
            print_json({"status": "configured", "context_config": str(context_path.resolve())}, stdout)
        else:
            print(f"窗口配置已保存：{context_path.resolve()}", file=stdout)
        return 0
    if args.interactive and (not is_terminal(stdin) or args.input == "-"):
        raise UsageError("--interactive 需要可读取回答的终端；请将需求放入 TEXT 或 --input 文件。")
    interactive = (not args.non_interactive and args.input != "-" and is_terminal(stdin)
                   and (args.interactive or is_terminal(stdout)))
    text = read_request(args, stdin, stderr, env_file)
    reference_paths = [check_input_path(path, env_file) for path in args.reference_file]
    history_paths = [check_input_path(path, env_file) for path in args.history_file]
    protected = [root / ".env", env_file, context_path, *reference_paths, *history_paths]
    if args.input and args.input != "-":
        protected.append(Path(args.input))
    if not args.preview_references:
        validate_dialogue_paths(root=root, output_path=args.output, report_path=args.report, protected_paths=protected)
    references = prepare_references([
        ReferenceFile(path, args.reference_purpose or DEFAULT_PURPOSE, args.reference_usage or DEFAULT_USAGE,
                      args.reference_kind) for path in reference_paths],
        ReferenceOptions(mode=args.reference_mode), query=text) if reference_paths else None
    if args.preview_references:
        if args.json:
            print_json(references.metadata(), stdout)
        else:
            print("\n\n".join(references.blocks), file=stdout)
            for warning in references.warnings:
                print(warning, file=stderr)
        return 0
    history = prepare_history(history_paths, text=args.history_text) if history_paths or args.history_text is not None else None
    configs = loader(env_file)
    windows = read_window_config(check_input_path(context_path, env_file)) if history is not None else None
    if windows is not None:
        WindowLimits.from_config(windows, configs)
    terminal = TerminalSession(args, stdin=stdin, stderr=stderr, root=root,
                               protected=protected, interactive=interactive)
    return asyncio.run(run_session(args, text=text, configs=configs, references=references,
        history=history, windows=windows, terminal=terminal, stdout=stdout, optimizer_factory=optimizer_factory))


def main(argv=None, *, stdin=None, stdout=None, stderr=None, root=ROOT,
         config_loader=None, optimizer_factory=None):
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    parser = build_parser(stdout=stdout, stderr=stderr)
    args = None
    try:
        args = parser.parse_args(argv)
        validate_arguments(args)
        return execute(args, stdin=stdin, stdout=stdout, stderr=stderr, root=Path(root).resolve(),
                       config_loader=config_loader, optimizer_factory=optimizer_factory)
    except SystemExit as error:
        return int(error.code)
    except KeyboardInterrupt:
        print("已取消。", file=stderr)
        return 130
    except BrokenPipeError:
        return 1
    except Exception as error:
        if isinstance(error, UsageError):
            message, code = str(error), 2
        elif isinstance(error, ImportError):
            message, code = "依赖未安装完整，请使用项目 .venv 或先安装 requirements.txt。", 1
        else:
            from optimizer_config import OptimizerError
            message = str(error) if isinstance(error, OptimizerError) else "运行失败，请检查输入、文件权限和项目环境。"
            code = 2 if isinstance(error, OptimizerError) else 1
        if args is not None and args.json:
            print_json({"status": "error", "reason": message, "exit_code": code}, stdout)
        print(f"错误：{message}", file=stderr)
        return code


if __name__ == "__main__":
    configure_stdio()
    raise SystemExit(main())
