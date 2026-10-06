"""Bounded CoT comparison; dry run by default, never prints credentials."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import load_cases
from evals.live_probe import Probe
from optimizer_io import configure_stdio
from optimizer_config import OptimizerError, load_models, read_environment
from optimizer_engine import BudgetExceeded
from optimizer_prompts import PROMPT_VERSION


PREVIOUS_VERSION = "2.3.0"
FIXTURES = {
    "cot-03": {
        "variables": {"{dimensions}": "长方形，长度 0.08 km，宽度 1200 cm。",
                      "{area_result}": "96 平方米"},
        "expected": "计算错误，正确面积为 960 平方米",
    },
    "cot-04": {
        "variables": {"{facts}": "每台进入灰度的设备都会收到标记。设备 R 收到了标记。没有其他前提。",
                      "{claims}": "设备 R 进入了灰度"},
        "expected": "UNSUPPORTED",
    },
    "cot-06": {
        "variables": {"{design_left}": "恢复时间 8 分钟，维护难度 3。",
                      "{design_right}": "恢复时间 11 分钟，维护难度 2。",
                      "{criteria}": "维护难度不超过 2 是硬性条件；符合条件的方案中，恢复时间越短越好。"},
        "expected": {"choice": "right"},
    },
}


def previous_templates():
    examples, namespace = {}, {}
    for stem, target in (("optimizer_examples", examples), ("optimizer_prompts", namespace)):
        path = ROOT / "baselines" / (stem + "_v2_3.py.txt")
        exec(compile(path.read_text(encoding="utf-8-sig"), str(path), "exec"), target)
    namespace["GENERATION_EXAMPLES"] = examples["GENERATION_EXAMPLES"]
    if namespace["PROMPT_VERSION"] != PREVIOUS_VERSION:
        raise ValueError("CoT 对照基线版本不匹配。")
    return namespace


def instantiate(prompt, fixture):
    for variable, value in fixture["variables"].items():
        prompt = prompt.replace(variable, value)
    return prompt


def check_response(case_id, response):
    text = response.strip()
    if case_id == "cot-03":
        return {"correct_verdict_and_area": bool(
            re.search(r"不正确|错误|有误|不对|不成立|未通过", text)
            and re.search(r"(?<!\d)960(?:\.0+)?(?!\d)", text)),
                "output_scope": "requires_manual_review"}
    if case_id == "cot-04":
        return {"correct_label_and_exact_format": text == "UNSUPPORTED"}
    if case_id == "cot-06":
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result

        try:
            result = json.loads(text, object_pairs_hook=unique_keys)
            valid = (isinstance(result, dict) and set(result) == {"choice"}
                     and result["choice"] == "right")
        except (ValueError, TypeError):
            valid = False
        return {"correct_choice_and_exact_json": valid}
    raise ValueError("没有该下游样例的校验规则。")


class CoTProbe(Probe):
    def __init__(self, destination, configs, cases, prior):
        super().__init__(destination, configs, request_limit=40,
                         token_limit=230000, elapsed_limit=1800)
        self.cases, self.prior = cases, prior
        self.data.update(
            scope="CoT prompt materials: paired development baselines, downstream answers, quality smoke",
            versions=[PREVIOUS_VERSION, PROMPT_VERSION],
            planned_requests=len(cases) * 2 + len(FIXTURES) * 3 + 3,
            fixtures=FIXTURES,
            comparison_controls={"same_baseline_model_and_parameters": True,
                                 "alternating_version_order": True,
                                 "downstream_json_mode_forced": False,
                                 "repeated_runs": 1,
                                 "quality_smoke_is_not_a_version_comparison": True},
        )

    def reserve(self, config, system, payload):
        if any(call["status"] != "running" and call.get("total_tokens") is None
               for call in self.data["calls"]):
            raise BudgetExceeded("已有请求的用量未知，停止后续真实调用。")
        return super().reserve(config, system, payload)

    async def downstream_case(self, case_id, version, prompt):
        self.label = "downstream/" + case_id + "/" + version
        fixture = FIXTURES[case_id]
        instantiated = instantiate(prompt, fixture)
        record = {"id": self.label, "input": instantiated,
                  "expected": fixture["expected"], "manual_review": None}
        self.data["records"].append(record)
        # Generation may force JSON, while one task explicitly requires a label.
        # All three prompt conditions use the same downstream transport setting.
        model = self.factory(replace(self.configs["a"], json_mode=False))
        try:
            reply = await asyncio.wait_for(model.complete(
                "请完成给定任务，遵守任务的最终输出格式；只返回任务答案。",
                {"task": instantiated, "phase": "downstream"}, timeout=90), timeout=95)
            record["response"] = reply.text
            record["deterministic_check"] = check_response(case_id, reply.text)
        except BudgetExceeded:
            raise
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": "未记录服务详情。"}
        finally:
            await model.close()
            self.checkpoint()

    async def run(self):
        comparisons = {}
        try:
            for index, case in enumerate(self.cases):
                versions = (PREVIOUS_VERSION, PROMPT_VERSION) if index % 2 == 0 else (PROMPT_VERSION, PREVIOUS_VERSION)
                for version in versions:
                    record = await self.optimize(case.id + "/" + version, case.input,
                                                 previous=self.prior if version == PREVIOUS_VERSION else None,
                                                 baseline=True)
                    record["case"] = case.model_dump()
                    result = record.get("result", {})
                    text = result.get("optimized_prompt") or ""
                    record["observations"] = {
                        "missing_literals": [item for item in case.required_literals if item not in text],
                        "clarification_matches_expectation": (
                            (result.get("status") == "needs_clarification") == case.expected_clarification
                            if result else None),
                        "semantic_quality": "requires_manual_review",
                    }
                    comparisons[(case.id, version)] = record
                    self.checkpoint()
            by_id = {case.id: case for case in self.cases}
            for index, case_id in enumerate(FIXTURES):
                prompts = {"original": by_id[case_id].input}
                for version in (PREVIOUS_VERSION, PROMPT_VERSION):
                    result = comparisons[(case_id, version)].get("result", {})
                    if result.get("status") == "unreviewed":
                        prompts[version] = result["optimized_prompt"]
                    else:
                        self.data["records"].append({"id": "downstream/" + case_id + "/" + version,
                                                     "skipped": "没有可执行的成功优化稿。"})
                order = ("original", PREVIOUS_VERSION, PROMPT_VERSION)
                order = order[index:] + order[:index]
                for version in order:
                    if version in prompts:
                        await self.downstream_case(case_id, version, prompts[version])
            await self.optimize("quality-smoke/" + PROMPT_VERSION, by_id["cot-02"].input)
            self.data["status"] = "complete"
        except BudgetExceeded as error:
            self.data.update(status="stopped_by_limit", stop_reason=str(error))
        finally:
            self.checkpoint()
            print(json.dumps({key: self.data[key] for key in (
                "status", "request_count", "known_total_tokens", "unknown_usage_requests", "elapsed_seconds")},
                ensure_ascii=False), flush=True)
            print("报告：" + str(self.destination), flush=True)


def main(argv=None):
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--load-env", action="store_true", help="显式允许程序加载 .env，不显示或保存配置")
    args = parser.parse_args(argv)
    cases = load_cases(ROOT / "evals" / "cot_reasoning.jsonl")
    prior = previous_templates()
    if not args.run:
        print(json.dumps({"dry_run": True, "versions": [PREVIOUS_VERSION, PROMPT_VERSION],
                          "cases": [case.id for case in cases], "downstream_cases": list(FIXTURES),
                          "planned_requests": len(cases) * 2 + len(FIXTURES) * 3 + 3,
                          "request_cap": 40, "known_tokens_soft_limit": 230000,
                          "note": "未读取配置或调用模型。"}, ensure_ascii=False, indent=2))
        return 0
    try:
        environment = read_environment(ROOT / ".env") if args.load_env else read_environment(None)
        configs = load_models(environment)
        del environment
        destination = ROOT / "evals" / "results" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-cot-probe.json")
        probe = CoTProbe(destination, configs, cases, prior)
        probe.data["configuration_source"] = "program-loaded env file" if args.load_env else "process environment"
        probe.data["snapshot_sha256"] = {
            stem: hashlib.sha256((ROOT / "baselines" / (stem + "_v2_3.py.txt")).read_bytes()).hexdigest()
            for stem in ("optimizer_prompts", "optimizer_examples")}
        asyncio.run(probe.run())
        return 0 if probe.data["status"] == "complete" else 4
    except OptimizerError as error:
        print("CoT 真实测试未启动或未完成：" + str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已取消，已发生请求的记录已保留。", file=sys.stderr)
        return 130
    except Exception as error:
        print("CoT 真实测试异常：" + type(error).__name__ + "；未输出服务详情。", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
