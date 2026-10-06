"""Targeted follow-up sharing the original CoT probe's total request budget."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import load_cases
from evals.cot_probe import CoTProbe
from evals.live_probe import Probe
from optimizer_io import configure_stdio
from optimizer_config import OptimizerError, load_models, read_environment
from optimizer_engine import BudgetExceeded
from optimizer_prompts import PROMPT_VERSION


SINGLE_CLAIM_REQUEST = (
    "判断单条命题 {claims} 是否能从 {facts} 严格推出。"
    "核对前提和结论，检查是否存在反例；能严格推出时输出 SUPPORTED，否则输出 UNSUPPORTED。"
    "最终只输出一个标签，不附理由。"
)


def remaining_limits(data):
    if data.get("status") != "complete" or data.get("unknown_usage_requests") != 0:
        raise ValueError("主对照未完整结束，或存在未知用量。")
    limits = data["limits"]
    remaining = {"request_limit": limits["requests"] - data["request_count"],
                 "token_limit": limits["known_tokens_soft_limit"] - data["known_total_tokens"],
                 "elapsed_limit": limits["elapsed_seconds"] - data["elapsed_seconds"]}
    if remaining["request_limit"] < 5 or remaining["token_limit"] <= 0 or remaining["elapsed_limit"] <= 0:
        raise ValueError("剩余预算不足以进行五次针对性核验。")
    return remaining


class FollowupProbe(CoTProbe):
    def __init__(self, destination, configs, baseline_path, baseline, cases):
        limits = remaining_limits(baseline)
        Probe.__init__(self, destination, configs,
                       request_limit=5, token_limit=limits["token_limit"],
                       elapsed_limit=limits["elapsed_limit"])
        self.cases = cases
        self.data.update(scope="CoT follow-up: exclusion in quality mode and unambiguous single claim",
                         baseline_report=str(baseline_path),
                         baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                         combined_request_cap=baseline["limits"]["requests"],
                         prior_request_count=baseline["request_count"],
                         prior_known_tokens=baseline["known_total_tokens"],
                         planned_requests=5,
                         prompt_version=PROMPT_VERSION,
                         fixture_note="单条命题任务沿用主对照的 cot-04 事实和反向蕴含期望结果；生成输入已显式消除多条断言汇总歧义。")

    async def run(self):
        try:
            await self.optimize("quality-exclude/" + PROMPT_VERSION, self.cases["cot-10"].input)
            record = await self.optimize("single-claim/" + PROMPT_VERSION, SINGLE_CLAIM_REQUEST, baseline=True)
            result = record.get("result", {})
            if result.get("status") == "unreviewed":
                await self.downstream_case("cot-04", PROMPT_VERSION, result["optimized_prompt"])
            else:
                self.data["records"].append({"id": "downstream/cot-04/" + PROMPT_VERSION,
                                             "skipped": "单条命题优化仍无可执行成功稿。"})
            self.data["status"] = "complete"
        except BudgetExceeded as error:
            self.data.update(status="stopped_by_limit", stop_reason=str(error))
        finally:
            self.checkpoint()
            print(json.dumps({"status": self.data["status"],
                              "request_count": self.data["request_count"],
                              "combined_request_count": self.data["prior_request_count"] + self.data["request_count"],
                              "known_total_tokens": self.data["known_total_tokens"],
                              "combined_known_tokens": self.data["prior_known_tokens"] + self.data["known_total_tokens"]},
                             ensure_ascii=False), flush=True)
            print("报告：" + str(self.destination), flush=True)


def main(argv=None):
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--load-env", action="store_true")
    args = parser.parse_args(argv)
    try:
        baseline_path = args.baseline_report.resolve()
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        limits = remaining_limits(baseline)
        if not args.run:
            print(json.dumps({"dry_run": True, "planned_requests": 5,
                              "prior_requests": baseline["request_count"],
                              "remaining_limits": limits}, ensure_ascii=False, indent=2))
            return 0
        environment = read_environment(ROOT / ".env") if args.load_env else read_environment(None)
        configs = load_models(environment)
        del environment
        destination = ROOT / "evals" / "results" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-cot-followup.json")
        cases = {case.id: case for case in load_cases(ROOT / "evals" / "cot_reasoning.jsonl")}
        probe = FollowupProbe(destination, configs, baseline_path, baseline, cases)
        asyncio.run(probe.run())
        return 0 if probe.data["status"] == "complete" else 4
    except (OptimizerError, ValueError) as error:
        print("CoT 追加核验未启动或未完成：" + str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已取消，已发生请求的记录已保留。", file=sys.stderr)
        return 130
    except Exception as error:
        print("CoT 追加核验异常：" + type(error).__name__ + "；未输出服务详情。", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
