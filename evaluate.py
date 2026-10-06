"""Fixed cases, explicit opt-in to model calls, and blank human review fields."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Literal

from pydantic import Field

from optimizer_io import atomic_write, configure_stdio
from optimizer_config import ROOT, ConfigurationError, OptimizerError, load_models, model_config, read_environment
from optimizer_engine import BudgetExceeded, Optimizer, RunOptions
from optimizer_models import OutputError, StrictModel, Text, parse_output
from optimizer_selection import DEFAULT_MAX_BUILTIN_EXAMPLES, DEFAULT_MAX_EXAMPLE_CHARS


class EvaluationCase(StrictModel):
    id: Text
    split: Literal["dev", "holdout"]
    input: Text
    expected_clarification: bool
    required_literals: list[Text]
    manual_checks: list[Text] = Field(min_length=1)


def load_cases(path: Path):
    cases = [parse_output(line, EvaluationCase) for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ConfigurationError("评测集不能为空，样例 ID 不能重复。")
    return cases


class EvaluationOptimizer(Optimizer):
    """Limit paid evaluation attempts independently of product sessions."""
    def __init__(self, configs, options, *, request_budget, **kwargs):
        super().__init__(configs, options, **kwargs)
        self._evaluation_request_budget = request_budget

    def _reserve(self, role, purpose, attempt=1):
        if len(self.calls) >= self._evaluation_request_budget:
            raise BudgetExceeded("本组开发评测的请求预算已用尽。")
        return super()._reserve(role, purpose, attempt)


class BaselineOptimizer(EvaluationOptimizer):
    """Reuse the archived v1 system template with the same bounded API adapter."""
    _required_model_roles = ("a",)
    _requires_independent_review = False

    def metadata(self):
        return super().metadata() | {"mode": "baseline", "prompt_version": "baseline-v1"}

    async def _workflow(self):
        template = (ROOT / "baselines" / "system_prompt_v1.txt").read_text(encoding="utf-8")
        reply = await self._call("a", template, {"original_request": self.original}, "baseline")
        if not reply.text.strip():
            raise OutputError("基线模型没有返回有效文本。")
        return self._finish("unreviewed", [], prompt=reply.text.strip(), reason="旧版系统模板，无独立评审。")


async def run_evaluation(cases, configs, options, call_budget, *, runner_factory=EvaluationOptimizer, checkpoint=None,
                         baseline=False):
    if type(call_budget) is not int or call_budget < 1:
        raise ConfigurationError("整组评测请求预算必须是正整数。")
    minimum = 1 if baseline else 3
    records = []
    used = 0
    for case in cases:
        remaining = call_budget - used
        if remaining < minimum:
            break
        runner = runner_factory(configs, options, request_budget=remaining)
        record = {"case": case.model_dump(), "case_sha256": hashlib.sha256(
            json.dumps(case.model_dump(), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
            "manual_review": {"intent_preserved": None, "constraints_preserved": None,
                              "no_invented_facts": None, "clarification_appropriate": None,
                              "preferred_over_baseline": None, "notes": ""}}
        try:
            result = await runner.run(case.input)
            text = result.optimized_prompt or ""
            record["result"] = result.to_dict()
            record["observations"] = {
                "missing_literals": [literal for literal in case.required_literals if literal not in text],
                "length_ratio": round(len(text) / len(case.input), 3),
                "clarification_matches_expectation": (
                    (result.status == "needs_clarification") == case.expected_clarification
                    if result.metadata.get("mode") != "baseline" and result.status != "needs_review" else None),
                "semantic_quality": "requires_human_review",
            }
        except OptimizerError as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        except Exception:
            record["error"] = {"type": "UnexpectedError", "message": "评测异常；未记录服务详情。"}
        record["metadata"] = runner.metadata()
        used += record["metadata"]["request_count"]
        records.append(record)
        if checkpoint is not None:
            checkpoint(records, used)
    return {"cases_requested": len(cases), "cases_completed": len(records),
            "cases_skipped_for_budget": len(cases) - len(records), "request_count": used,
            "records": records, "semantic_quality": "requires_human_review"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="固定样例评测；默认只检查样例，不读取 .env 或调用模型")
    parser.add_argument("--cases", type=Path, default=ROOT / "evals" / "cases.jsonl")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--mode", choices=("baseline", "quality"), default="quality")
    parser.add_argument("--run", action="store_true", help="明确启用真实模型请求")
    parser.add_argument("--call-budget", type=int, default=12, help="整组请求尝试数上限")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-builtin-examples", type=int, default=DEFAULT_MAX_BUILTIN_EXAMPLES)
    parser.add_argument("--max-example-chars", type=int, default=DEFAULT_MAX_EXAMPLE_CHARS)
    parser.add_argument("--output", type=Path)
    env = parser.add_mutually_exclusive_group()
    env.add_argument("--env-file", type=Path)
    env.add_argument("--no-env", action="store_true")
    args = parser.parse_args(argv)
    latest = None
    try:
        if args.limit < 1 or args.call_budget < 1:
            raise ConfigurationError("样例数量和请求预算必须为正整数。")
        options = RunOptions(retries=args.retries,
                             max_builtin_examples=args.max_builtin_examples, max_example_chars=args.max_example_chars)
        minimum = 1 if args.mode == "baseline" else 3
        if args.call_budget < minimum:
            raise ConfigurationError("请求预算不足以完成一个样例的基本流程。")
        cases = [c for c in load_cases(args.cases) if args.split == "all" or c.split == args.split][:args.limit]
        if not cases:
            raise ConfigurationError("所选分组没有样例。")
        if not args.run:
            print(json.dumps({"dry_run": True, "cases": [c.id for c in cases], "mode": args.mode,
                              "minimum_requests_without_failures": len(cases) * minimum,
                              "total_request_cap": args.call_budget,
                              "note": "未读取配置或调用模型；追加 --run 才会实际运行。"}, ensure_ascii=False, indent=2))
            return 0
        env_path = None if args.no_env else (args.env_file or ROOT / ".env")
        destination = args.output or ROOT / "evals" / "results" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + args.mode + ".json")
        if destination.resolve() in {p.resolve() for p in (args.cases, env_path) if p is not None}:
            raise ConfigurationError("评测报告不能覆盖样例或配置文件。")
        environment = read_environment(env_path)
        configs = {"a": model_config(environment, "GENERATOR_A", "a")} if args.mode == "baseline" else load_models(environment)
        del environment

        def checkpoint(records, used):
            nonlocal latest
            latest = {"mode": args.mode, "split": args.split, "request_count": used, "records": records}
            atomic_write(destination, json.dumps(latest, ensure_ascii=False, indent=2))
            print(f"已记录 {len(records)}/{len(cases)} 个样例，请求 {used}/{args.call_budget}。", file=sys.stderr)

        result = asyncio.run(run_evaluation(cases, configs, options, args.call_budget,
                            runner_factory=BaselineOptimizer if args.mode == "baseline" else EvaluationOptimizer,
                            checkpoint=checkpoint, baseline=args.mode == "baseline"))
        latest = {"mode": args.mode, "split": args.split, **result}
        atomic_write(destination, json.dumps(latest, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in latest.items() if k != "records"}, ensure_ascii=False, indent=2))
        print(f"报告：{destination.resolve()}；语义质量需人工复核。", file=sys.stderr)
        return 4 if result["cases_skipped_for_budget"] or any("error" in r for r in result["records"]) else 0
    except (ConfigurationError, OutputError) as error:
        print(str(error), file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        if latest is not None:
            print(json.dumps(latest, ensure_ascii=False, indent=2))
        print("评测文件读写失败；已获得的记录见标准输出。", file=sys.stderr)
        return 5
    except KeyboardInterrupt:
        print("评测已取消，已完成样例保留在报告中。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    configure_stdio()
    raise SystemExit(main())
