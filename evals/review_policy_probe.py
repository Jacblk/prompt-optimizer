"""Re-review saved real candidates with current criteria; dry-run by default."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.live_probe import Probe, safe_metadata
from optimizer_io import configure_stdio
from optimizer_config import OptimizerError, load_models, read_environment
from optimizer_engine import Candidate, Optimizer, RunOptions
from optimizer_models import Draft
from optimizer_prompts import PROMPT_VERSION
from optimizer_review import require_review_backend


def cached_candidates(data):
    record = next(record for record in data["records"] if record["id"] == "cot-10/quality")
    candidates = [Candidate(item["id"], item["origin"],
                            Draft.model_validate({key: item[key] for key in Draft.model_fields}))
                  for item in record["result"]["candidates"]]
    if (len(candidates) != 3 or {candidate.origin for candidate in candidates} != {"a", "b", "original"}
            or next(candidate for candidate in candidates if candidate.origin == "original").draft.optimized_prompt
            != record["input"]):
        raise ValueError("缓存候选不符合三候选复核要求。")
    return record, candidates


class CachedReviewOptimizer(Optimizer):
    async def _workflow(self):
        # Generation is deliberately omitted: these drafts came from saved real calls.
        return self._decision(await self._review(self.cached, "cached_review"), self.cached)


async def run(source_path, data, quality, destination):
    source_record, candidates = cached_candidates(data)
    probe = Probe(destination, quality, request_limit=1, token_limit=24000, elapsed_limit=300)
    probe.label = "cot-10/cached-policy-review"
    probe.data.update(scope="one real official review of previously generated real candidates; no new generation",
                      prompt_version=PROMPT_VERSION, review_backend=require_review_backend(),
                      source_report=str(source_path), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                      source_prompt_version=source_record["result"]["metadata"]["prompt_version"],
                      planned_requests=1, test_judge_timeout_seconds=180,
                      configuration_policy="program-only .env loading; no configuration values saved")
    record = {"id": probe.label, "input": source_record["input"],
              "cached_candidates": [candidate.to_dict() for candidate in candidates]}
    probe.data["records"].append(record)
    runner = CachedReviewOptimizer(quality, RunOptions(allow_repair=False, retries=0), factory=probe.factory)
    runner.cached = candidates
    try:
        result = await runner.run(source_record["input"])
        record["result"] = result.to_dict()
        record["result"]["metadata"] = safe_metadata(result.metadata)
        record["checks"] = {"one_real_judge_request": len(probe.data["calls"]) == 1,
                            "explicit_exclusion_retained": result.status == "ready" and
                            "不要添加思维链或分步处理要求" in (result.optimized_prompt or ""),
                            "known_usage": runner.unknown_usage == 0}
        probe.data["status"] = "complete" if all(record["checks"].values()) else "completed_with_errors"
    except OptimizerError as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        probe.data["status"] = "error"
    finally:
        record["metadata"] = safe_metadata(runner.metadata())
        probe.checkpoint()
        print(json.dumps({key: probe.data[key] for key in
                          ("status", "request_count", "known_total_tokens", "unknown_usage_requests", "elapsed_seconds")},
                         ensure_ascii=False), flush=True)
        print("报告：" + str(destination), flush=True)
    return 0 if probe.data["status"] == "complete" else 4


def main(argv=None):
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--load-env", action="store_true")
    args = parser.parse_args(argv)
    try:
        source = args.source_report.resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        cached_candidates(data)
        if not args.run:
            print(json.dumps({"dry_run": True, "planned_requests": 1, "generation_requests": 0,
                              "prompt_version": PROMPT_VERSION}, ensure_ascii=False))
            return 0
        require_review_backend()
        environment = read_environment(ROOT / ".env" if args.load_env else None)
        quality = load_models(environment)
        del environment
        quality["judge"] = replace(quality["judge"], timeout=180)
        destination = ROOT / "evals" / "results" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-review-policy-probe.json")
        return asyncio.run(run(source, data, quality, destination))
    except (OptimizerError, ValueError, KeyError, StopIteration, OSError) as error:
        print("缓存候选复核未完成：" + type(error).__name__ + "；未输出配置或服务详情。", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已取消，已发生请求的记录已保留。", file=sys.stderr)
        return 130
    except Exception as error:
        print("缓存候选复核异常：" + type(error).__name__ + "；未输出配置或服务详情。", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
