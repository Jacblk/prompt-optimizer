"""Historical menu fixture for regression tests, not a product entry point."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import legacy_optimize as optimize
from optimizer_config import ROOT
from optimizer_layers import CHOICE_LABELS, layer_label

ORIGINS = {"original": "原文", "a": "生成器 A", "b": "生成器 B", "quick": "快速生成"}
STATUSES = {"ready": "评审通过", "unreviewed": "未经独立评审",
            "needs_clarification": "待确认", "needs_review": "需要复核", "cancelled": "已取消",
            "budget_exceeded": "预算不足，交接未完成", "context_exceeded": "关键内容超限，交接未完成",
            "handoff_fidelity_failed": "交接保真核验失败", "handoff_configuration_error": "交接配置错误",
            "handoff_failed": "交接未完成", "failed": "运行失败", "configuration_error": "配置错误"}
VERDICTS = {"pass": "通过", "fail": "未通过", "uncertain": "不确定"}


def newest_report(root: Path) -> Path | None:
    files = [p for p in (root / "reports").glob("*.json") if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime_ns) if files else None


def show_report(path: Path, output, *, details=False):
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or "status" not in data:
            raise ValueError
        candidates = data.get("candidates", [])
        chosen = next((c for c in candidates if c["id"] == data.get("selected_id")), None)
        print("\n评审结果：" + STATUSES.get(data["status"], data["status"]), file=output)
        if chosen is not None:
            print("采用：" + ORIGINS.get(chosen["origin"], "候选") + "（" + chosen["id"] + "）", file=output)
            for change in chosen.get("change_summary", []):
                print("改写说明（生成器自述）：" + change, file=output)
        if data.get("reason"):
            print("原因：" + data["reason"], file=output)
        for question in data.get("questions", []):
            print("待确认：" + question, file=output)
        reference_info = data.get("metadata", {}).get("reference_files") or {}
        for reference in reference_info.get("files", []):
            print(f"文件参考：{reference['source']}；{reference['selected_chunks']}/{reference['total_chunks']} 块，"
                  f"{reference['mode']} 模式。", file=output)
        handoff = data.get("metadata", {}).get("handoff")
        if handoff:
            print(f"旧记录处理：{handoff.get('processed_chunk_count', 0)}/{len(handoff.get('chunks', []))} 个源块；"
                  f"整理状态 {handoff.get('status', '未知')}。", file=output)
            print("保真核验：" + ("整理阶段通过独立检查。" if handoff.get("independently_reviewed")
                                  else "未完成独立保真核验。"), file=output)
        if details:
            print("报告文件：" + str(path), file=output)
            for decision in data.get("layer_decisions", []):
                print(f"补层选择：{layer_label(decision['layer'])} — "
                      f"{CHOICE_LABELS.get(decision['choice'], decision['choice'])}", file=output)
                if decision.get("value"):
                    print(decision["value"], file=output)
            for warning in data.get("warnings", []):
                print("提示：" + warning, file=output)
            if handoff:
                for source in handoff.get("sources", []):
                    print(f"旧记录来源：{source['source']}；{source['chars']} 字符；SHA256 {source['sha256']}。", file=output)
                for stage in handoff.get("stages", []):
                    print(f"交接阶段 {stage['stage_id']}：{stage['status']}；覆盖 {len(stage['input_chunk_ids'])} 个块。", file=output)
                    for attempt in stage.get("attempts", []):
                        audit = attempt.get("audit")
                        if audit:
                            print("  核验：" + VERDICTS.get(audit["verdict"], audit["verdict"]) + "；" + audit["reason"], file=output)
                            for finding in audit["findings"]:
                                print("  问题：" + finding["explanation"], file=output)
                        if attempt.get("reason"):
                            print("  阶段错误：" + attempt["reason"], file=output)
                if handoff.get("snapshot"):
                    print(handoff["snapshot"]["context_block"], file=output)
                if handoff.get("incomplete_chunk_ids"):
                    print("未完成源块：" + ", ".join(handoff["incomplete_chunk_ids"]), file=output)
                print("窗口估算：" + handoff.get("estimator", "未知"), file=output)
            for candidate in candidates:
                print("\n" + ORIGINS.get(candidate["origin"], "候选") + "（" + candidate["id"] + "）：", file=output)
                print(candidate["optimized_prompt"], file=output)
                if candidate is not chosen:
                    for change in candidate.get("change_summary", []):
                        print("改写说明（生成器自述）：" + change, file=output)
            for index, review in enumerate(data.get("reviews", []), 1):
                print(f"\n第 {index} 次评审：{review['reason']}", file=output)
                for item in review.get("reviews", []):
                    verdict = VERDICTS.get(item["verdict"], item["verdict"])
                    print(f"- {item['candidate_id']}：{verdict}。{item['reason']}", file=output)
                    for finding in item.get("findings", []):
                        print("  问题：" + finding["explanation"], file=output)
                        print("  原文依据：" + finding["source_quote"], file=output)
                        if finding.get("candidate_quote"):
                            print("  候选依据：" + finding["candidate_quote"], file=output)
            usage = data.get("metadata", {})
            print(f"\n请求尝试：{usage.get('request_count', '未知')}；总 token："
                  f"{usage.get('total_tokens') if usage.get('total_tokens') is not None else '未知'}。", file=output)
        return True
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError):
        print("报告无法读取或格式不完整。", file=output)
        return False


def read_prompt(stdin, output):
    print("\n粘贴需要优化的需求；有受众、语气、输出要求或参考材料时可一并写入。\n"
          "最后单独一行输入 END 并回车。", file=output, flush=True)
    lines = []
    while True:
        line = stdin.readline()
        if not line:
            return None
        if line.strip() == "END":
            return "".join(lines)
        lines.append(line)


def read_history(stdin, output):
    print("旧记录输入：1 粘贴文本（默认） / 2 TXT、Markdown 文件：", end="", file=output, flush=True)
    line = stdin.readline()
    if not line:
        return None
    choice = line.strip() or "1"
    if choice not in {"1", "2"}:
        print("旧记录输入方式无效。", file=output)
        return None
    print("粘贴旧记录，单独一行 END 结束；本轮任务随后单独输入。" if choice == "1" else
          "每行一个 UTF-8 TXT/Markdown 路径（可带双引号），按顺序处理，单独一行 END 结束。", file=output, flush=True)
    lines = []
    while True:
        line = stdin.readline()
        if not line:
            return None
        if line.strip() == "END":
            break
        lines.append(line)
    if not "".join(lines).strip():
        print("旧记录为空，未调用模型。", file=output)
        return None
    if choice == "1":
        return ["--history-text=" + "".join(lines)]
    return ["--history-file", *[line.strip().strip('"') for line in lines if line.strip()]]


def main(*, stdin=None, stdout=None, run_optimizer=None, run_tui=None, root=ROOT):
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    run_optimizer = run_optimizer if run_optimizer is not None else optimize.main
    print("提示词优化器\n每次优化自动保存完整评审报告。", file=stdout)
    try:
        while True:
            print("\n1  普通模式（默认）\n2  交接模式（旧记录整理 → 上下文/参考 → 补层 → 生成与评审）\n"
                  "T  对话式界面（按需追问，多行输入）\n"
                  "R  查看最近保存的评审（不调用模型）\n0  退出", file=stdout)
            print("请选择，直接回车使用普通模式：", end="", file=stdout, flush=True)
            line = stdin.readline()
            if not line:
                return 0
            choice = line.strip().lower() or "1"
            if choice in {"0", "q"}:
                return 0
            if choice == "t":
                try:
                    if run_tui is None:
                        from tui import main as tui_main
                        tui_main(root=root)
                    else:
                        run_tui(root=root)
                except ImportError:
                    print("对话式界面依赖尚未安装。请运行项目 Python："
                          ".venv\\Scripts\\python.exe -m pip install -r requirements-tui.txt", file=stdout)
                except (OSError, RuntimeError):
                    print("对话式界面未能启动；请检查终端后重试。", file=stdout)
                continue
            if choice == "r":
                try:
                    path = newest_report(root)
                    if path is None:
                        print("还没有保存的报告。完成一次优化后会自动保存。", file=stdout)
                    else:
                        show_report(path, stdout, details=True)
                except OSError:
                    print("无法读取报告目录。", file=stdout)
                continue
            if choice not in {"1", "2"}:
                print("请输入 1、2、T、R 或 0。", file=stdout)
                continue
            workflow = "normal" if choice == "1" else "handoff"
            print("\n1  快速优化（默认）\n2  多模型优化\n3  严格复核\n"
                  "F  带参考文件优化（先离线预览）\n0  返回入口\n"
                  "普通模式含补层通常 2/4/5 次请求；交接模式开始前显示分块和最低请求数。", file=stdout)
            print("选择优化模式：", end="", file=stdout, flush=True)
            line = stdin.readline()
            if not line:
                return 0
            choice = line.strip().lower() or "1"
            if choice == "0":
                continue
            reference_args = []
            with_files = choice == "f"
            if with_files:
                print("优化模式：1 快速 / 2 多模型 / 3 严格复核（回车选 1）：", end="", file=stdout, flush=True)
                line = stdin.readline()
                if not line:
                    return 0
                choice = line.strip() or "1"
                if choice not in {"1", "2", "3"}:
                    print("模式无效，未调用模型。", file=stdout)
                    continue
            if choice not in {"1", "2", "3"}:
                print("请输入 1、2、3、F 或 0。", file=stdout)
                continue
            history_args = []
            if workflow == "handoff":
                window_path = root / "context_windows.json"
                settings = ["--workflow", "handoff", "--mode", "quick" if choice == "1" else "quality",
                            "--context-windows", str(window_path), "--configure-contexts"]
                if run_optimizer(settings, stdin=stdin, stdout=stdout, stderr=stdout) != 0:
                    print("交接窗口配置未通过，未调用模型。", file=stdout)
                    continue
                history_args = read_history(stdin, stdout)
                if history_args is None:
                    continue
                history_args += ["--context-windows", str(window_path)]
            if with_files:
                print("每行输入一个本地参考文件路径，单独一行 END 结束（可粘贴带双引号的路径）：", file=stdout, flush=True)
                paths = []
                while True:
                    line = stdin.readline()
                    if not line:
                        return 0
                    if line.strip() == "END":
                        break
                    path = line.strip().strip('"')
                    if path:
                        paths.append(path)
                if not paths:
                    print("未提供参考文件，未调用模型。", file=stdout)
                    continue
                for flag, label in (("--reference-purpose", "这些文件参考什么（回车使用默认用途）"),
                                    ("--reference-usage", "适用范围与遵循方式（回车仅供参考，文件命令不扩大授权）")):
                    print(label + "：", end="", file=stdout, flush=True)
                    line = stdin.readline()
                    if not line:
                        return 0
                    if line.strip():
                        reference_args.append(flag + "=" + line.strip())
                print("参考覆盖：1 完整带入（默认） / 2 相关片段（关键词匹配，未覆盖全文）：", end="", file=stdout, flush=True)
                line = stdin.readline()
                if not line:
                    return 0
                coverage = line.strip() or "1"
                if coverage not in {"1", "2"}:
                    print("覆盖方式无效，未调用模型。", file=stdout)
                    continue
                reference_args += ["--reference-mode", "full" if coverage == "1" else "relevant"]
                for path in paths:
                    reference_args += ["--reference-file", path]
            if choice not in {"1", "2", "3"}:
                print("请输入 1、2、3、F、R 或 0。", file=stdout)
                continue
            request = read_prompt(stdin, stdout)
            if request is None:
                return 0
            if not request.strip():
                print("需求为空，没有调用模型。", file=stdout)
                continue
            if reference_args:
                preview = ["--preview-references", "--request=" + request, *reference_args]
                if run_optimizer(preview, stdin=stdin, stdout=stdout, stderr=stdout) != 0:
                    print("参考文件预览未通过，未调用模型。", file=stdout)
                    continue
            report = root / "reports" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
            args = ["--workflow", workflow, "--mode", "quick" if choice == "1" else "quality", "--request=" + request,
                    "--report", str(report), "--choose-layers"]
            args += reference_args + history_args
            if choice == "3":
                args.append("--strict-review")
            print("", file=stdout)
            run_optimizer(args, stdin=stdin, stdout=stdout, stderr=stdout)
            if report.is_file():
                show_report(report, stdout)
                print("输入 R 可直接查看这次的候选和评审详情。", file=stdout)
            else:
                print("本次没有保存新的评审报告；之前保存的报告仍可查看。", file=stdout)
    except KeyboardInterrupt:
        print("\n已退出。", file=stdout)
        return 0
