"""Historical CLI fixture for regression tests, not a product entry point."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from optimizer_io import atomic_write, configure_stdio

from optimizer_config import ROOT, ConfigurationError, OptimizerError, load_models, read_environment
from optimizer_engine import OptimizationResult, Optimizer, RunOptions
from optimizer_prompts import PROMPT_VERSION
from optimizer_layers import prompt_for_layers
from optimizer_selection import DEFAULT_MAX_BUILTIN_EXAMPLES, DEFAULT_MAX_EXAMPLE_CHARS
from optimizer_documents import DEFAULT_PURPOSE, DEFAULT_USAGE, ReferenceFile, ReferenceOptions, prepare_references
from optimizer_handoff import WindowLimits, model_identity, prepare_history, read_window_config, required_roles


def make_parser():
    parser = argparse.ArgumentParser(description="保留原意的提示词优化器")
    parser.add_argument("--workflow", choices=("normal", "handoff"), default="normal")
    parser.add_argument("--mode", choices=("quick", "quality"), default="quality")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--request", help="直接传入原始需求")
    source.add_argument("--input-file", type=Path, help="UTF-8 文本文件")
    history = parser.add_mutually_exclusive_group()
    history.add_argument("--history-file", type=Path, nargs="+", action="extend", default=[],
                         help="交接旧记录，UTF-8 TXT/Markdown；多份按参数顺序处理")
    history.add_argument("--history-text", help="粘贴旧记录；与当前 --request 独立")
    parser.add_argument("--context-windows", type=Path, help="公开角色窗口配置，默认 context_windows.json")
    parser.add_argument("--configure-contexts", action="store_true", help="填写并绑定交接角色窗口，不调用模型")
    parser.add_argument("--context-window", action="append", default=[], metavar="ROLE=SIZE",
                        help="配合 --configure-contexts 显式填写窗口，例如 quick=128000；未提供的缺失角色交互填写")
    env = parser.add_mutually_exclusive_group()
    env.add_argument("--env-file", type=Path, help="指定配置文件；默认项目目录 .env")
    env.add_argument("--no-env", action="store_true", help="仅使用进程环境变量")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=Path, help="最终提示词保存路径")
    output.add_argument("--no-save", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--report", type=Path, help="可选完整 JSON 报告，包含候选和原始引用")
    parser.add_argument("--strict-review", action="store_true")
    parser.add_argument("--choose-layers", action=argparse.BooleanOptionalAction, default=None,
                        help="交互终端默认开启：识别缺失层，选择自行增加、模型补全或省略")
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--retries", type=int, default=1, help="每次逻辑调用的额外暂时错误重试数")
    parser.add_argument("--token-budget", type=int, help="按已报告用量停止后续请求的软预算")
    parser.add_argument("--max-input-chars", type=int, default=50000)
    parser.add_argument("--max-builtin-examples", type=int, default=DEFAULT_MAX_BUILTIN_EXAMPLES,
                        help="内置教学示范数量上限；0 表示关闭，不影响用户参考材料")
    parser.add_argument("--max-example-chars", type=int, default=DEFAULT_MAX_EXAMPLE_CHARS,
                        help="内置教学示范 JSON 块字符上限；整例省略，不截断内容")
    parser.add_argument("--reference-file", type=Path, action="append", default=[],
                        help="明确加载的本地参考文件，可重复；与任务输入 --input-file 分开")
    parser.add_argument("--reference-purpose", default=DEFAULT_PURPOSE, help="这些文件参考什么")
    parser.add_argument("--reference-usage", default=DEFAULT_USAGE, help="适用范围与遵循方式")
    parser.add_argument("--reference-kind", choices=("auto", "implementation", "artifact", "execution"), default="auto")
    parser.add_argument("--reference-mode", choices=("full", "relevant"), default="full",
                        help="默认完整带入；relevant 显式选择本地关键词匹配片段，不保证全文覆盖")
    parser.add_argument("--reference-chunk-size", type=int, default=1600)
    parser.add_argument("--reference-chunk-overlap", type=int, default=160)
    parser.add_argument("--max-reference-chars", type=int, default=20000, help="文件参考块合计字符上限，包含来源与范围")
    parser.add_argument("--max-reference-chunks", type=int, default=6, help="relevant 模式整块数量上限；full 不使用此上限")
    parser.add_argument("--preview-references", action="store_true", help="离线预览加载与分块，不读取模型配置或调用模型")
    parser.add_argument("--check-config", action="store_true", help="只检查配置，不发起模型请求")
    return parser


def read_request(args, stdin, stderr):
    if args.request is not None:
        request = args.request
    elif args.input_file is not None:
        if any(p.name.casefold().startswith(".env") for p in (args.input_file, args.input_file.resolve())):
            raise ConfigurationError("配置文件 .env 不能作为任务文本读取。")
        request = args.input_file.read_text(encoding="utf-8-sig")
    elif not stdin.isatty():
        request = stdin.read()
    else:
        print("请输入原始需求，单独一行 END 结束（文件或管道输入不使用结束标记）：", file=stderr)
        lines = []
        while True:
            line = stdin.readline()
            if not line or line.strip() == "END":
                break
            lines.append(line)
        request = "".join(lines)
    if not request.strip():
        raise ConfigurationError("原始需求不能为空。")
    return request


def render_text(result):
    if result.status in {"ready", "unreviewed"}:
        return result.optimized_prompt
    title = {"needs_clarification": "待确认", "cancelled": "已取消", "budget_exceeded": "预算不足，交接未完成",
             "context_exceeded": "关键内容超限，交接未完成", "handoff_fidelity_failed": "交接保真核验失败",
             "handoff_configuration_error": "交接配置错误", "handoff_failed": "交接未完成"}.get(result.status, "需要复核")
    parts = [title, result.reason]
    parts.extend(f"- {question}" for question in result.questions)
    if result.optimized_prompt:
        parts.extend(["以下为待确认草稿：", result.optimized_prompt])
    return "\n\n".join(part for part in parts if part)


def configure_context_windows(configs, mode, path, settings, stdin, output):
    """Explicit limits, bound to public model/service hashes, never guessed."""
    data = read_window_config(path) if path.exists() else {"version": 1, "roles": {}}
    fields = ("model", "service_sha256", "identity_sha256", "context_window")
    roles = {role: {k: v for k, v in entry.items() if k in fields}
             for role, entry in data["roles"].items() if role in {"quick", "a", "b", "judge"} and isinstance(entry, dict)}
    explicit = {}
    for value in settings:
        try:
            role, size = value.split("=", 1)
            if role not in required_roles() or role in explicit:
                raise ValueError
            explicit[role] = int(size)
        except ValueError:
            raise ConfigurationError("--context-window 必须为当前角色=正整数，且角色不能重复。") from None
    for role in required_roles():
        identity = model_identity(configs[role])
        entry = roles.get(role, {})
        valid = all(entry.get(k) == v for k, v in identity.items()) and type(entry.get("context_window")) is int
        size = explicit.get(role)
        if size is None and valid:
            size = entry["context_window"]
        if size is None:
            print(f"角色 {role}：{configs[role].name}；服务摘要 {identity['service_sha256'][:12]}；"
                  f"输出上限 {configs[role].max_tokens}。", file=output)
            print("请按服务实际限制填写上下文窗口 token 上限（0 或输入结束取消）：", end="", file=output, flush=True)
            value = stdin.readline()
            try:
                size = int(value.strip())
            except ValueError:
                raise ConfigurationError("未填写有效窗口，未修改窗口配置。") from None
        if size < 1:
            raise ConfigurationError("窗口填写已取消，未修改窗口配置。")
        roles[role] = {**identity, "context_window": size}
    result = {"version": 1, "roles": roles}
    WindowLimits.from_config(result, configs)
    atomic_write(path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"角色窗口已核对并保存：{path.resolve()}；不含密钥，未调用模型。", file=output)


def main(argv=None, *, stdin=None, stdout=None, stderr=None, optimizer_factory=Optimizer):
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    args = make_parser().parse_args(argv)
    handoff = args.workflow == "handoff"
    report_path = args.report
    if handoff and report_path is None and not (args.check_config or args.configure_contexts or args.preview_references):
        report_path = ROOT / "reports" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-handoff.json")
    report_safe = False
    history_bundle = None
    if args.choose_layers is None:
        args.choose_layers = stdin.isatty()
    try:
        if not handoff and (args.history_file or args.history_text is not None or args.context_windows is not None
                            or args.configure_contexts or args.context_window):
            raise ConfigurationError("历史及窗口参数仅用于 --workflow handoff。")
        if args.context_window and not args.configure_contexts:
            raise ConfigurationError("--context-window 需要 --configure-contexts。")
        options = RunOptions(
                              allow_repair=not args.no_repair, retries=args.retries,
                             token_budget=args.token_budget, max_input_chars=args.max_input_chars,
                             choose_layers=args.choose_layers, max_builtin_examples=args.max_builtin_examples,
                             max_example_chars=args.max_example_chars)
        reference_options = ReferenceOptions(mode=args.reference_mode, chunk_size=args.reference_chunk_size,
                                             chunk_overlap=args.reference_chunk_overlap,
                                             max_chars=args.max_reference_chars, max_chunks=args.max_reference_chunks)
        reference_files = [ReferenceFile(path, args.reference_purpose, args.reference_usage, args.reference_kind)
                           for path in args.reference_file]
        env_path = None if args.no_env else (args.env_file or ROOT / ".env")
        protected_config = (env_path or ROOT / ".env").resolve()
        window_path = (args.context_windows or ROOT / "context_windows.json").expanduser().resolve() if handoff else None
        material_paths = [p.expanduser().resolve() for p in (*args.reference_file, *args.history_file, args.input_file,
                                                          window_path) if p is not None]
        if any(p == protected_config or (p.exists() and protected_config.exists() and p.samefile(protected_config))
               for p in material_paths):
            raise ConfigurationError("模型配置文件不能作为任务文本或参考材料读取。")
        if args.preview_references and (handoff or args.check_config or args.output is not None or args.report is not None):
            raise ConfigurationError("离线参考预览不能同时检查模型配置或保存优化结果/评审报告。")
        if args.configure_contexts:
            if (args.check_config or args.output is not None or args.report is not None or args.history_file
                    or args.history_text is not None or args.request is not None or args.input_file is not None or args.reference_file):
                raise ConfigurationError("填写窗口是独立操作，不能同时读取任务/旧记录或保存优化报告。")
            if window_path.name.casefold().startswith(".env"):
                raise ConfigurationError("窗口配置不能覆盖 .env。")
            configs = load_models(read_environment(env_path))
            configure_context_windows(configs, args.mode, window_path, args.context_window, stdin, stdout)
            return 0
        if args.check_config:
            configs = load_models(read_environment(env_path))
            if handoff:
                WindowLimits.from_config(read_window_config(window_path), configs)
            print("配置结构检查通过；尚未验证 API 连接或模型可用性。", file=stdout)
            return 0
        if args.preview_references:
            if not reference_files:
                raise ConfigurationError("参考预览需要至少一个 --reference-file。")
            request = read_request(args, stdin, stderr) if (args.reference_mode == "relevant"
                       or args.request is not None or args.input_file is not None) else ""
            bundle = prepare_references(reference_files, reference_options, query=request)
            if request and len(request) + bundle.chars > options.max_input_chars:
                raise ConfigurationError("原始需求与文件参考合计超过本次允许的字符数。")
            if args.format == "json":
                print(json.dumps(bundle.metadata(), ensure_ascii=False, indent=2), file=stdout)
            else:
                print(f"离线参考预览：{len(bundle.files)} 份文件，{bundle.chars} 字符；未调用模型。", file=stdout)
                print("\n\n".join(bundle.blocks), file=stdout)
            for warning in bundle.warnings:
                print("提示：" + warning, file=stderr)
            return 0
        output_path = None if args.no_save else (args.output or ROOT / "last_optimized_prompt.md")
        destinations = [p.resolve() for p in (output_path, report_path) if p is not None]
        inputs = [p.resolve() for p in (env_path, args.input_file) if p is not None] + material_paths
        same_file = any(a.exists() and b.exists() and a.samefile(b) for a in destinations for b in inputs)
        if (len(set(destinations)) != len(destinations) or set(destinations).intersection(inputs) or same_file
                or any(p.name.casefold().startswith(".env") for p in destinations)):
            raise ConfigurationError("结果、报告路径不能互相重合，也不能覆盖输入、参考或配置文件。")
        last_success = ROOT / "last_optimized_prompt.md"
        if handoff and report_path is not None and (report_path.resolve() == last_success.resolve()
                or report_path.exists() and last_success.exists() and report_path.samefile(last_success)):
            raise ConfigurationError("交接阶段报告不能覆盖上次成功提示词，包括 --no-save 时。")
        report_safe = True
        if handoff:
            history_paths = [p.expanduser().resolve() for p in args.history_file]
            current_paths = [p.expanduser().resolve() for p in (args.input_file, *args.reference_file) if p is not None]
            if any(old == current or (old.exists() and current.exists() and old.samefile(current))
                   for old in history_paths for current in current_paths):
                raise ConfigurationError("旧记录不能同时作为本轮任务文件或原始参考文件；请独立提供当前需求和必要参考。")
            history_bundle = prepare_history(args.history_file, text=args.history_text)
        request = read_request(args, stdin, stderr)
        if len(request) > options.max_input_chars:
            raise ConfigurationError("原始需求超过本次允许的字符数。")
        bundle = prepare_references(reference_files, reference_options, query=request) if reference_files else None
        if bundle is not None:
            if len(request) + bundle.chars > options.max_input_chars:
                raise ConfigurationError("原始需求与文件参考合计超过本次允许的字符数。")
            for file in bundle.files:
                print(f"参考文件：{file.source}；{len(file.selected)}/{len(file.chunks)} 块，{args.reference_mode} 模式。", file=stderr)
            for warning in bundle.warnings:
                print("提示：" + warning, file=stderr)
        configs = load_models(read_environment(env_path))
        if args.choose_layers:
            print("先整理/核验旧记录，再识别尚缺的提示层；已有上下文和参考无需重复提供。" if handoff else
                  "先识别提示词缺失层（通常增加 1 次模型请求），再根据你的选择生成。", file=stderr, flush=True)
        print("正在调用……", file=stderr)
        extra = {"layer_resolver": lambda analysis: prompt_for_layers(analysis, stdin, stderr)} if args.choose_layers else {}
        if bundle is not None:
            extra["references"] = bundle
        if handoff:
            extra.update(history=history_bundle, context_windows=window_path,
                         progress=lambda message: print(message, file=stderr, flush=True))
        result = asyncio.run(optimizer_factory(configs, options, **extra).run(request))
        text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2) if args.format == "json" else render_text(result)
        # Deliver the generated text before attempting any disk writes.
        print(text, file=stdout, flush=True)
        for warning in result.warnings:
            print("提示：" + warning, file=stderr)
        if result.status == "unreviewed":
            print("快速模式结果未经独立模型评审。", file=stderr)
        writes = []
        if report_path is not None:
            writes.append((report_path, json.dumps(result.to_dict(), ensure_ascii=False, indent=2)))
        if output_path is not None and result.status in {"ready", "unreviewed"}:
            writes.append((output_path, result.optimized_prompt))
        save_failed = False
        for path, content in writes:
            try:
                atomic_write(path, content)
                print(f"已保存：{path.resolve()}", file=stderr)
            except (OSError, UnicodeError):
                save_failed = True
                print("文件保存失败；生成结果已输出到标准输出。", file=stderr)
        count = result.metadata.get("request_count", 0)
        tokens = result.metadata.get("total_tokens")
        print(f"请求尝试：{count}；总 token：{tokens if tokens is not None else '未知'}。", file=stderr)
        if save_failed:
            return 5
        if result.status == "cancelled":
            return 130
        return 0 if result.status in {"ready", "unreviewed"} else 4
    except ConfigurationError as error:
        if handoff and report_safe:
            save_preparation_failure(report_path, str(error), history_bundle, args, stdout, stderr)
        print(f"配置或输入错误：{error}", file=stderr)
        return 2
    except OptimizerError as error:
        if handoff and report_safe:
            save_preparation_failure(report_path, str(error), history_bundle, args, stdout, stderr)
        print(f"运行失败：{error}", file=stderr)
        return 3
    except (OSError, UnicodeError):
        if handoff and report_safe:
            save_preparation_failure(report_path, "文件或终端读写失败，交接未完成。", history_bundle, args, stdout, stderr)
        print("文件或终端读写失败；请检查路径、权限和 UTF-8 编码。", file=stderr)
        return 5
    except KeyboardInterrupt:
        if handoff and report_safe:
            save_preparation_failure(report_path, "输入已取消，交接未完成。", history_bundle, args, stdout, stderr, status="cancelled")
        print("操作已取消。", file=stderr)
        return 130
    except Exception:
        print("运行失败：发生未预期错误，未输出可能包含凭据的服务详情。", file=stderr)
        return 3


def save_preparation_failure(path, reason, history, args, stdout, stderr, *, status="handoff_configuration_error"):
    result = OptimizationResult(status, None, False, reason=reason, metadata={
        "workflow": "handoff", "mode": args.mode, "prompt_version": PROMPT_VERSION, "request_count": 0,
        "calls": [], "total_tokens": 0, "handoff": {"status": "not_started", "run_status": status,
        "sources": [s.summary() for s in history.sources] if history else [], "chunks": [], "stages": [],
        "requested_sources": [str(path.expanduser().resolve()) for path in args.history_file]
        if args.history_file else (["粘贴旧记录"] if args.history_text is not None else []),
        "snapshot": None, "raw_history_saved": False, "incomplete": True}})
    if args.format == "json":
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), file=stdout, flush=True)
    try:
        atomic_write(path, json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        print(f"交接阶段报告已保存：{path.resolve()}", file=stderr)
    except (OSError, UnicodeError):
        print("交接阶段报告保存失败。", file=stderr)
