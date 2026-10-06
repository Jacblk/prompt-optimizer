"""Bounded compatibility check for the official evaluator; dry-run by default."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import load_cases
from evals.live_probe import Probe
from optimizer_io import configure_stdio
from optimizer_config import OptimizerError, load_models, read_environment
from optimizer_engine import BudgetExceeded
from optimizer_prompts import PROMPT_VERSION
from optimizer_review import require_review_backend


PLAN = ("cot-02", "cot-10", "cot-06")
REQUEST_LIMIT = len(PLAN) * 3


class ReviewProbe(Probe):
    def __init__(self, destination, quality, cases, *, plan=PLAN, judge_timeout=None, workflow_timeout=240):
        self.plan, self.workflow_timeout = plan, workflow_timeout
        request_limit = len(plan) * 3
        super().__init__(destination, quality, request_limit=request_limit,
                         token_limit=80000, elapsed_limit=900)
        self.cases = cases
        self.data.update(scope="official OpenEvals compatibility using current CoT criteria",
                         prompt_version=PROMPT_VERSION, planned_requests=request_limit,
                         test_overrides={"judge_timeout_seconds": judge_timeout,
                                         "workflow_timeout_seconds": workflow_timeout},
                         review_backend=require_review_backend(),
                         dependencies={name: version(name) for name in
                                       ("langchain", "langchain-core", "langchain-openai", "langsmith", "openevals")},
                         source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                        for name in ("optimizer_review.py", "optimizer_engine.py", "optimizer_models.py")},
                         configuration_policy="program-only .env loading; no configuration values saved",
                         automatic_cloud_tracing=False)

    def reserve(self, config, system, payload):
        if any(call["status"] != "running" and call.get("total_tokens") is None
               for call in self.data["calls"]):
            raise BudgetExceeded("存在未知用量，停止后续兼容性请求。")
        return super().reserve(config, system, payload)

    async def run(self):
        try:
            for case_id in self.plan:
                case = self.cases[case_id]
                record = await self.optimize(case_id + "/quality", case.input,
                                             total_timeout=self.workflow_timeout)
                record["manual_checks"] = case.manual_checks
                result = record.get("result", {})
                metadata = record["metadata"]
                judge_calls = [call for call in metadata["calls"] if call["role"] == "judge"]
                record["compatibility_checks"] = {
                    "structured_result": bool(result),
                    "official_backend": metadata.get("review_backend") == self.data["review_backend"],
                    "expected_request_count": metadata["request_count"] == 3,
                    "one_request_per_review": len(judge_calls) == 1,
                    "review_history_complete": len(result.get("reviews", [])) == 1,
                    "all_request_usage_known": metadata["unknown_usage_requests"] == 0,
                    "roles_reused": {call["role"] for call in metadata["calls"]} == {"a", "b", "judge"},
                }
                self.checkpoint()
            passed = all(all(record["compatibility_checks"].values()) for record in self.data["records"])
            self.data["status"] = "complete" if passed else "completed_with_errors"
        except BudgetExceeded as error:
            self.data.update(status="stopped_by_limit", stop_reason=str(error))
        finally:
            self.checkpoint()
            print(json.dumps({key: self.data[key] for key in
                              ("status", "request_count", "known_total_tokens", "unknown_usage_requests", "elapsed_seconds")},
                             ensure_ascii=False), flush=True)
            print("报告：" + str(self.destination), flush=True)


def main(argv=None):
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--load-env", action="store_true", help="仅程序加载配置，不展示或记录配置值")
    parser.add_argument("--cases", nargs="+", choices=list(PLAN), default=list(PLAN))
    parser.add_argument("--judge-timeout", type=float, help="仅覆盖本次测试的评审等待秒数，不修改配置文件")
    parser.add_argument("--workflow-timeout", type=float, default=240)
    args = parser.parse_args(argv)
    try:
        if len(args.cases) != len(set(args.cases)):
            raise ValueError("测试样例不能重复。")
        for seconds in (args.judge_timeout, args.workflow_timeout):
            if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
                raise ValueError("测试超时必须为有限的正数。")
        plan = tuple(case_id for case_id in PLAN if case_id in args.cases)
        request_limit = len(plan) * 3
        cases = {case.id: case for case in load_cases(ROOT / "evals" / "cot_reasoning.jsonl")}
        if not args.run:
            print(json.dumps({"dry_run": True, "backend": "openevals", "prompt_version": PROMPT_VERSION,
                              "planned_requests": request_limit, "cases": list(plan),
                              "roles": ["a", "b", "judge"]},
                             ensure_ascii=False, indent=2))
            return 0
        require_review_backend()
        environment = read_environment(ROOT / ".env" if args.load_env else None)
        quality = load_models(environment)
        del environment
        if args.judge_timeout is not None:
            quality["judge"] = replace(quality["judge"], timeout=args.judge_timeout)
        destination = ROOT / "evals" / "results" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-review-probe.json")
        probe = ReviewProbe(destination, quality, cases, plan=plan, judge_timeout=args.judge_timeout,
                            workflow_timeout=args.workflow_timeout)
        asyncio.run(probe.run())
        return 0 if probe.data["status"] == "complete" else 4
    except (OptimizerError, ValueError) as error:
        print("评审兼容性测试未完成：" + str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已取消，已发生请求的记录已保留。", file=sys.stderr)
        return 130
    except Exception as error:
        print("评审兼容性测试异常：" + type(error).__name__ + "；未输出服务详情。", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
