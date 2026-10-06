"""Bounded 2.9.2/2.9.3 comparison; dry-run unless --run is explicit."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.layer_boundary_cases import CASES
from evals.live_probe import Probe, safe_metadata
from optimizer_config import OptimizerError, load_models, read_environment
from optimizer_engine import BudgetExceeded, Candidate, Optimizer, RunOptions
from optimizer_io import configure_stdio
from optimizer_models import Draft
from optimizer_prompts import PROMPT_VERSION
from optimizer_review import require_review_backend

BASELINE = ROOT / "baselines" / "layer_boundary_20261005" / "templates.json"
LIMITS = {"requests": 36, "tokens": 400000, "elapsed_seconds": 1200, "request_seconds": 180}


def load_baseline(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    expected = json.loads(json.dumps([asdict(case) for case in CASES], ensure_ascii=False))
    if data["prompt_version"] != "2.9.2" or data["cases"] != expected:
        raise ValueError("Baseline version or held-out cases changed")
    for case in CASES:
        for role in ("a", "b"):
            entry = data["generation"][case.id][role]
            if hashlib.sha256(entry["system"].encode("utf-8")).hexdigest() != entry["sha256"]:
                raise ValueError("Generation baseline checksum mismatch")
    if hashlib.sha256(data["review"].encode("utf-8")).hexdigest() != data["review_sha256"]:
        raise ValueError("Review baseline checksum mismatch")
    return data


def ready_draft(text):
    return Draft(status="ready", optimized_prompt=text, preserved_constraints=[],
                 clarification_questions=[], change_summary=[])


def fixed_candidates(case):
    candidates = [Candidate("", "a", ready_draft(case.good)),
                  Candidate("", "original", ready_draft(case.request))]
    if case.bad is not None:
        candidates.append(Candidate("", "b", ready_draft(case.bad)))
    random.Random(42).shuffle(candidates)
    gold = {}
    for index, candidate in enumerate(candidates, 1):
        candidate.id = f"candidate_{index}"
        gold[candidate.id] = "fail" if candidate.origin == "b" else "pass"
    return candidates, gold


def surface_observations(case, text):
    """Literal observations only; these do not certify semantic correctness."""
    headings = list(re.finditer(r"(?m)^## (指令层|情境层|参考层|输出层)\s*\n", text))
    sections = {match[1]: text[match.end():headings[index + 1].start() if index + 1 < len(headings) else len(text)]
                for index, match in enumerate(headings)}
    return {"headings": [match[1] for match in headings],
            "materials": [{"literal": literal, "expected_primary_layer": layer,
                           "occurrences": text.count(literal),
                           "observed_layers": [name for name, content in sections.items() if literal in content]}
                          for literal, layer in case.material_roles],
            "semantic_quality": "requires_independent_source_review"}


def gold_scores(review, gold):
    actual = {item["candidate_id"]: item["verdict"] for item in review["reviews"]}
    return {"negative_count": sum(value == "fail" for value in gold.values()),
            "misses": sum(actual[key] != "fail" for key, value in gold.items() if value == "fail"),
            "positive_count": sum(value == "pass" for value in gold.values()),
            "false_flags": sum(actual[key] != "pass" for key, value in gold.items() if value == "pass"),
            "expected": gold, "actual": actual}


def probe_metadata(metadata):
    result = safe_metadata(metadata)
    for call in result.get("calls", []):
        call.pop("returned_model", None)
    return result


def recover_reported_usage(data):
    """Recover numeric adapter evidence only; unknown usage remains unknown."""
    recovered = []
    by_id = {record["id"]: record for record in data["records"]}
    for call in data["calls"]:
        if call.get("total_tokens") is not None or call["status"] != "error":
            continue
        recorded = by_id.get(call["test"], {}).get("metadata", {}).get("calls", [])
        if len(recorded) == 1 and type(recorded[0].get("total_tokens")) is int and recorded[0]["total_tokens"] >= 0:
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = recorded[0].get(key)
                call[key] = value if type(value) is int and value >= 0 else None
            call["usage_recovered_from_adapter"] = True
            recovered.append(call["index"])
    data["usage_recovery"] = recovered
    return data


class GenerationOnly(Optimizer):
    _requires_independent_review = False

    async def _workflow(self):
        draft = await self._generate(self.probe_role)
        candidate = Candidate("generation_candidate", self.probe_role, draft)
        if draft.status == "needs_clarification":
            return self._finish("needs_clarification", [candidate], questions=draft.clarification_questions,
                                reason="Generation requested clarification; not treated as a usable draft.")
        return self._finish("unreviewed", [candidate], prompt=draft.optimized_prompt,
                            selected=candidate.id, reason="Held-out generation comparison; no judge scoring.")


class FixedReview(Optimizer):
    async def _workflow(self):
        return self._decision(await self._review(self.fixed, "boundary_fixture_review"), self.fixed)


class BoundaryProbe(Probe):
    def __init__(self, destination, configs, baseline, resume=None):
        super().__init__(destination, configs, request_limit=LIMITS["requests"],
                         token_limit=LIMITS["tokens"], elapsed_limit=LIMITS["elapsed_seconds"])
        self.baseline = baseline
        self.data.update(scope="held-out generation and fixed-candidate review; prompt changes only",
                         before_version="2.9.2", after_version=PROMPT_VERSION,
                         planned_requests=36, planned_generation_requests=24, planned_review_requests=12,
                         request_timeout_seconds=180, automatic_cloud_tracing=False,
                         configuration_policy="program-only .env loading; no configuration values saved",
                         review_backend=require_review_backend(),
                         template_baseline_sha256=hashlib.sha256(json.dumps(baseline, ensure_ascii=False,
                                                                            sort_keys=True).encode("utf-8")).hexdigest())
        if resume is not None:
            if (resume["before_version"] != "2.9.2" or resume["after_version"] != PROMPT_VERSION
                    or resume["template_baseline_sha256"] != self.data["template_baseline_sha256"]
                    or any(type(call.get("total_tokens")) is not int or call["total_tokens"] < 0
                           for call in resume["calls"])):
                raise ValueError("Resume requires matching templates and known usage for every previous request")
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(resume["started_at_utc"])).total_seconds()
            self.started = time.monotonic() - max(age, resume["elapsed_seconds"])
            self.data = resume
            self.data["status"] = "running"

    def factory(self, config):
        recorded = super().factory(config)
        owner = self

        class MeteredModel:
            async def complete(self, *args, **kwargs):
                index = len(owner.data["calls"])
                try:
                    return await recorded.complete(*args, **kwargs)
                except OptimizerError as error:
                    usage = getattr(error, "usage", None)
                    if isinstance(usage, dict) and len(owner.data["calls"]) > index:
                        for key in ("input_tokens", "output_tokens", "total_tokens"):
                            value = usage.get(key)
                            owner.data["calls"][index][key] = value if type(value) is int and value >= 0 else None
                        owner.checkpoint()
                    raise

            def disable_streaming(self):
                recorded.disable_streaming()

            async def close(self):
                await recorded.close()

        return MeteredModel()

    def checkpoint(self):
        # The reusable recorder also receives the provider's returned model ID;
        # this probe retains roles only, never configuration/model identifiers.
        for call in self.data["calls"]:
            call.pop("returned_model", None)
        super().checkpoint()

    def reserve(self, config, system, payload):
        if any(call["status"] != "running" and call.get("total_tokens") is None for call in self.data["calls"]):
            raise BudgetExceeded("存在未知用量，已停止后续对照请求。")
        limits = self.data["limits"]
        if (len(self.data["calls"]) >= limits["requests"]
                or sum(call.get("total_tokens") or 0 for call in self.data["calls"]) >= limits["known_tokens_soft_limit"]
                or time.monotonic() - self.started >= limits["elapsed_seconds"]):
            raise BudgetExceeded("真实对照达到请求、用量或总等待上限。")
        entry = {"index": len(self.data["calls"]) + 1, "test": self.label, "role": config.role,
                 "phase": "review" if "candidates" in payload else "generate",
                 "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
                 "system_characters": len(system), "payload": payload, "status": "running"}
        self.data["calls"].append(entry)
        self.checkpoint()
        print(f"请求 {entry['index']}/{limits['requests']} 开始：{self.label} / {config.role} / {entry['phase']}", flush=True)
        return entry

    async def step(self, case, version, phase, role=None):
        self.label = f"{case.id}/{version}/{phase}" + (f"/{role}" if role else "")
        record = {"id": self.label, "case_id": case.id, "version": version, "phase": phase,
                  "role": role or "judge", "input": case.request, "rubric": list(case.rubric),
                  "independent_source_review": None}
        self.data["records"].append(record)
        options = RunOptions(allow_repair=False, retries=0, token_budget=LIMITS["tokens"])
        runner = (GenerationOnly if phase == "generation" else FixedReview)(
            self.configs, options, factory=self.factory, rng=random.Random(42))
        if phase == "generation":
            runner.probe_role = role
        else:
            runner.fixed, gold = fixed_candidates(case)
            record["gold_verdicts"] = gold
            record["fixed_candidates"] = [candidate.to_dict() for candidate in runner.fixed]
        with ExitStack() as stack:
            if version == "2.9.2":
                if phase == "generation":
                    def archived(strategy, *_args, **_kwargs):
                        entry = self.baseline["generation"][case.id][strategy]
                        return entry["system"], entry["selection"]
                    stack.enter_context(patch("optimizer_engine.prepare_generation_prompt", archived))
                else:
                    stack.enter_context(patch("optimizer_engine.review_prompt", lambda: self.baseline["review"]))
                stack.enter_context(patch("optimizer_engine.PROMPT_VERSION", "2.9.2"))
            try:
                result = await runner.run(case.request)
                record["result"] = result.to_dict()
                record["result"]["metadata"] = probe_metadata(result.metadata)
                if phase == "generation":
                    record["surface_observations"] = surface_observations(case, result.optimized_prompt or "")
                else:
                    record["gold_scores"] = gold_scores(result.reviews[-1], gold)
            except (BudgetExceeded, TimeoutError):
                record["error"] = {"type": "StoppedByLimit", "message": "预算、等待上限或未知用量；保留已完成记录。"}
                raise
            except OptimizerError as error:
                record["error"] = {"type": type(error).__name__}
            except Exception as error:
                record["error"] = {"type": type(error).__name__}
            finally:
                record["metadata"] = probe_metadata(runner.metadata())
                self.checkpoint()
        if any(call["status"] != "running" and call.get("total_tokens") is None for call in self.data["calls"]):
            raise BudgetExceeded("本次调用用量未知，停止后续对照请求。")

    async def run(self):
        try:
            remaining = max(0, LIMITS["elapsed_seconds"] - (time.monotonic() - self.started))
            async with asyncio.timeout(remaining):
                for index, case in enumerate(CASES):
                    versions = ("2.9.2", PROMPT_VERSION) if index % 2 == 0 else (PROMPT_VERSION, "2.9.2")
                    for role in ("a", "b"):
                        for version in versions:
                            label = f"{case.id}/{version}/generation/{role}"
                            if not any(record["id"] == label for record in self.data["records"]):
                                await self.step(case, version, "generation", role)
                    for version in versions:
                        label = f"{case.id}/{version}/review"
                        if not any(record["id"] == label for record in self.data["records"]):
                            await self.step(case, version, "review")
            self.data["status"] = ("complete" if len(self.data["records"]) == 36
                                   and all("result" in record for record in self.data["records"])
                                   else "completed_with_errors")
        except (BudgetExceeded, TimeoutError):
            self.data.update(status="stopped_by_limit", stop_reason="请求、用量、等待上限或未知用量。")
        finally:
            self.data["review_summary"] = {
                version: {key: sum(record.get("gold_scores", {}).get(key, 0)
                                  for record in self.data["records"] if record["version"] == version)
                          for key in ("negative_count", "misses", "positive_count", "false_flags")}
                for version in ("2.9.2", PROMPT_VERSION)}
            self.checkpoint()
            print(json.dumps({key: self.data[key] for key in
                              ("status", "request_count", "known_total_tokens", "unknown_usage_requests", "elapsed_seconds")},
                             ensure_ascii=False), flush=True)
            print("报告：" + str(self.destination), flush=True)
        return 0 if self.data["status"] == "complete" else 4


def main(argv=None):
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-templates", type=Path, default=BASELINE)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--load-env", action="store_true")
    parser.add_argument("--resume-report", type=Path, help="Continue unattempted steps with the original global budgets")
    args = parser.parse_args(argv)
    try:
        baseline = load_baseline(args.before_templates)
        resume = (recover_reported_usage(json.loads(args.resume_report.read_text(encoding="utf-8")))
                  if args.resume_report else None)
        if not args.run:
            print(json.dumps({"dry_run": True, "cases": [case.id for case in CASES], "planned_requests": 36,
                              "generation_requests": 24, "review_requests": 12, "limits": LIMITS,
                              "before_version": "2.9.2", "after_version": PROMPT_VERSION}, ensure_ascii=False))
            return 0
        environment = read_environment(ROOT / ".env" if args.load_env else None)
        configs = {role: replace(config, timeout=180) for role, config in load_models(environment).items()}
        del environment
        destination = args.resume_report if resume is not None else ROOT / "evals" / "results" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-layer-boundary-probe.json")
        return asyncio.run(BoundaryProbe(destination, configs, baseline, resume=resume).run())
    except KeyboardInterrupt:
        print("已取消，已发生的请求记录保留。", file=sys.stderr)
        return 130
    except Exception as error:
        print("四层对照未完成：" + type(error).__name__ + "；未输出配置或服务详情。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
