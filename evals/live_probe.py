"""Bounded real-model probe; explicit --run and optional --load-env are required.

No credentials/configuration values are printed or saved. Old/new comparison
changes prompt materials only; every model response comes from the real adapter.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from functools import partial
import json
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import load_cases
from optimizer_io import atomic_write, configure_stdio
from optimizer_config import OptimizerError, load_models, read_environment
from optimizer_engine import BudgetExceeded, Candidate, Optimizer, RunOptions
from optimizer_layers import LayerDecision
from optimizer_models import LangChainChatModel
from optimizer_prompts import PROMPT_VERSION


def previous_templates():
    examples = {}
    path = ROOT / "baselines" / "optimizer_examples_v2_2_1.py.txt"
    exec(compile(path.read_text(encoding="utf-8-sig"), str(path), "exec"), examples)
    namespace = {}
    path = ROOT / "baselines" / "optimizer_prompts_v2_2_1.py.txt"
    exec(compile(path.read_text(encoding="utf-8-sig"), str(path), "exec"), namespace)
    # The archived module imports today's examples. Explicitly restore its own
    # example snapshot before using its prompt-building functions.
    namespace["GENERATION_EXAMPLES"] = examples["GENERATION_EXAMPLES"]
    return namespace


def safe_metadata(metadata):
    result = {k: v for k, v in metadata.items() if k != "models"}
    result["calls"] = [{k: v for k, v in call.items() if k != "model"}
                       for call in result.get("calls", [])]
    return result


def prepare_archived_generation(previous, strategy, *_args, baseline=False, **_kwargs):
    """Keep historical fixed examples and report their actual sizes/IDs."""
    examples = previous["GENERATION_EXAMPLES"]
    return previous["generation_prompt"]("quick" if baseline else strategy), {
        "selector": "archived-fixed", "example_ids": [
            f"{previous['PROMPT_VERSION']}:example_{i}" for i in range(1, len(examples) + 1)],
        "example_count": len(examples),
        "example_characters": len(json.dumps(examples, ensure_ascii=False)),
        "available_example_count": len(examples),
        "all_example_characters": len(json.dumps(examples, ensure_ascii=False)),
        "max_builtin_examples": None, "max_example_chars": None, "matched_features": [],
    }


class DraftBaselineOptimizer(Optimizer):
    """Development-only single-generator comparison of prompt materials."""
    _required_model_roles = ("a",)
    _requires_independent_review = False

    def metadata(self):
        return super().metadata() | {"mode": "baseline"}

    async def _workflow(self):
        draft = await self._generate("a")
        candidate = Candidate("baseline_candidate", "a", draft)
        if draft.status == "needs_clarification":
            return self._finish("needs_clarification", [candidate], questions=draft.clarification_questions,
                                reason="开发对照中的单生成器仍需澄清。")
        return self._finish("unreviewed", [candidate], prompt=draft.optimized_prompt,
                            selected=candidate.id, reason="开发单模型对照，无独立评审。")


class Probe:
    def __init__(self, destination, configs, *, request_limit=20,
                 token_limit=160000, elapsed_limit=1200):
        self.destination = destination
        self.configs = configs
        self.started = time.monotonic()
        self.label = ""
        self.data = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "real models; prompt-material comparison on current engine",
            "limits": {"requests": request_limit, "known_tokens_soft_limit": token_limit,
                       "elapsed_seconds": elapsed_limit, "retries": 0},
            "records": [], "calls": [], "status": "running",
        }

    def checkpoint(self):
        self.data["request_count"] = len(self.data["calls"])
        self.data["known_total_tokens"] = sum(c.get("total_tokens") or 0 for c in self.data["calls"])
        self.data["unknown_usage_requests"] = sum(
            c.get("total_tokens") is None for c in self.data["calls"] if c["status"] != "running")
        self.data["elapsed_seconds"] = round(time.monotonic() - self.started, 2)
        atomic_write(self.destination, json.dumps(self.data, ensure_ascii=False, indent=2))

    def reserve(self, config, system, payload):
        limits = self.data["limits"]
        if (len(self.data["calls"]) >= limits["requests"]
                or sum(c.get("total_tokens") or 0 for c in self.data["calls"]) >= limits["known_tokens_soft_limit"]
                or time.monotonic() - self.started >= limits["elapsed_seconds"]):
            raise BudgetExceeded("真实测试已达到整轮请求、用量或等待上限。")
        if sum(c["status"] == "error" for c in self.data["calls"]) >= 2:
            raise BudgetExceeded("真实接口已出现两次调用错误，停止后续探测。")
        entry = {"index": len(self.data["calls"]) + 1, "test": self.label, "role": config.role,
                 "phase": payload.get("phase", "repair" if "repair" in payload else
                                      "review" if "candidates" in payload else "generate"),
                 "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
                 "system_characters": len(system), "payload": payload, "status": "running"}
        self.data["calls"].append(entry)
        self.checkpoint()
        print(f"请求 {entry['index']}/{limits['requests']} 开始：{self.label} / {config.role} / {entry['phase']}", flush=True)
        return entry

    def factory(self, config):
        owner = self
        real = LangChainChatModel(config)

        class RecordedModel:
            def disable_streaming(self):
                real.disable_streaming()

            async def complete(self, system, payload, *, timeout, on_activity=None):
                entry = owner.reserve(config, system, payload)
                started = time.monotonic()
                try:
                    # Paid probes keep their own wait cap, independently of the
                    # product's unlimited response read and inactivity warning.
                    async with asyncio.timeout(timeout):
                        reply = await real.complete(system, payload, timeout=timeout, on_activity=on_activity)
                    entry.update(status="ok", response=reply.text, input_tokens=reply.input_tokens,
                                 output_tokens=reply.output_tokens, total_tokens=reply.total_tokens,
                                 returned_model=reply.returned_model)
                    return reply
                except asyncio.CancelledError:
                    entry["status"] = "cancelled"
                    raise
                except Exception as error:
                    # Class names only; never echo provider errors/configuration.
                    entry.update(status="error", error_type=type(error).__name__)
                    raise
                finally:
                    entry["elapsed_seconds"] = round(time.monotonic() - started, 2)
                    owner.checkpoint()
                    print(f"请求 {entry['index']}/{owner.data['limits']['requests']} 结束：{entry['status']}，{entry['elapsed_seconds']} 秒", flush=True)

            async def close(self):
                await real.close()

        return RecordedModel()

    async def optimize(self, label, original, *, previous=None, baseline=False, layer_choice=None,
                       total_timeout=240):
        self.label = label
        record = {"id": label, "input": original, "manual_review": None}
        self.data["records"].append(record)
        options = RunOptions(choose_layers=layer_choice is not None,
                             retries=0, allow_repair=False)

        def choose(analysis):
            return [LayerDecision(layer=layer.layer,
                                  choice="model" if layer.layer == "references" and layer_choice == "model" else "omit",
                                  value=layer.suggestion if layer.layer == "references" and layer_choice == "model" else "")
                    for layer in analysis.missing()]

        runner_type = DraftBaselineOptimizer if baseline else Optimizer
        configs = {"a": self.configs["a"]} if baseline else self.configs
        runner = runner_type(configs, options, factory=self.factory, rng=random.Random(42),
                             layer_resolver=choose if layer_choice else None)
        # Only prompt builders/version are replaced for the archived comparison;
        # no fake model, canned output or synthetic usage is involved.
        with ExitStack() as stack:
            if previous:
                stack.enter_context(patch("optimizer_engine.prepare_generation_prompt",
                                          partial(prepare_archived_generation, previous, baseline=baseline)))
                for name in ("review_prompt", "PROMPT_VERSION"):
                    stack.enter_context(patch("optimizer_engine." + name, previous[name]))
            try:
                # This timeout belongs to the opt-in paid probe, not the optimizer.
                try:
                    async with asyncio.timeout(total_timeout):
                        result = await runner.run(original)
                except TimeoutError:
                    raise BudgetExceeded("开发探针已达到本样例的等待时间上限。") from None
                record["result"] = result.to_dict()
                record["result"]["metadata"] = safe_metadata(result.metadata)
                print(f"样例完成：{label}，状态 {result.status}", flush=True)
            except OptimizerError as error:
                record["error"] = {"type": type(error).__name__, "message": str(error)}
                print(f"样例未完成：{label}，{type(error).__name__}", flush=True)
                if isinstance(error, BudgetExceeded):
                    raise
            except Exception as error:
                record["error"] = {"type": type(error).__name__, "message": "未记录服务详情。"}
                print(f"样例异常：{label}，{type(error).__name__}", flush=True)
            finally:
                record["metadata"] = safe_metadata(runner.metadata())
                self.checkpoint()
        return record

    async def downstream(self, label, prompt):
        self.label = label
        instantiated = (prompt.replace("{proposal}", "排查时先清空线上缓存，再重启服务。")
                        .replace("{requirements}", "只读排查，不允许清空缓存或重启服务。"))
        record = {"id": label, "input": instantiated, "expected_approved": False, "manual_review": None}
        self.data["records"].append(record)
        model = self.factory(self.configs["judge"])
        try:
            reply = await asyncio.wait_for(model.complete(
                "请完成用户给出的任务，遵守其中的最终输出格式。",
                {"task": instantiated, "phase": "downstream"}, timeout=90), timeout=95)
            record["response"] = reply.text
            try:
                value = json.loads(reply.text)
                passed = (isinstance(value, dict) and set(value) == {"approved", "note"}
                          and value["approved"] is False
                          and isinstance(value["note"], str) and bool(value["note"].strip()))
            except (ValueError, TypeError):
                passed = False
            record["deterministic_check"] = {"correct_verdict_and_json_shape": passed,
                                             "note_semantics": "requires_manual_review"}
        except OptimizerError as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            if isinstance(error, BudgetExceeded):
                raise
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": "未记录服务详情。"}
        finally:
            await model.close()
            self.checkpoint()

    async def run(self):
        cases = {c.id: c for c in load_cases(ROOT / "evals" / "few_shot_steps.jsonl")}
        prior = previous_templates()
        comparisons = {}
        try:
            for index, suffix in enumerate(("02", "03", "08", "10")):
                case = cases["shots-steps-" + suffix]
                # Alternate order to reduce a systematic before/after ordering bias.
                versions = ("2.2.1", PROMPT_VERSION) if index % 2 == 0 else (PROMPT_VERSION, "2.2.1")
                for version in versions:
                    record = await self.optimize(case.id + "/" + version, case.input,
                                                 previous=prior if version == "2.2.1" else None, baseline=True)
                    record["manual_checks"] = case.manual_checks
                    comparisons[(suffix, version)] = record
                    self.checkpoint()
            await self.optimize("interactive-omit/" + PROMPT_VERSION, "你是谁", layer_choice="omit")
            await self.optimize("interactive-model/" + PROMPT_VERSION,
                                "这是可复用分类提示：检查 {message} 是否含有“错误”二字；含有则输出 ERROR，否则输出 OK。"
                                "最终只输出一个标签，不附带解释。", layer_choice="model", baseline=True)
            await self.downstream("downstream/original", cases["shots-steps-10"].input)
            for version in ("2.2.1", PROMPT_VERSION):
                result = comparisons[("10", version)].get("result", {})
                if result.get("status") in {"ready", "unreviewed"}:
                    await self.downstream("downstream/" + version, result["optimized_prompt"])
                else:
                    self.data["records"].append({"id": "downstream/" + version,
                                                 "skipped": "优化阶段没有可用的成功结果。"})
            self.data["status"] = "complete"
        except BudgetExceeded:
            self.data["status"] = "stopped_by_limit"
        finally:
            self.checkpoint()
            print(json.dumps({k: self.data[k] for k in ("status", "request_count", "known_total_tokens",
                                                       "unknown_usage_requests", "elapsed_seconds")}, ensure_ascii=False), flush=True)
            print("报告：" + str(self.destination), flush=True)


def main():
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--load-env", action="store_true", help="仅程序加载 .env；不打印或保存配置")
    args = parser.parse_args()
    if not args.run:
        print("未读取配置或调用模型；--run 才会真实调用。整轮最多 20 次请求。")
        return 0
    try:
        # Only this program accesses the file; raw configuration is never returned
        # to the assistant, printed, added to chat payloads, or written to reports.
        environment = read_environment(ROOT / ".env" if args.load_env else None)
        configs = load_models(environment)
        del environment
        destination = ROOT / "evals" / "results" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-live-probe.json")
        probe = Probe(destination, configs)
        asyncio.run(probe.run())
        return 0 if probe.data["status"] == "complete" else 4
    except OptimizerError as error:
        print("真实测试未启动：" + str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已取消，已完成请求的报告已保留。", file=sys.stderr)
        return 130
    except Exception as error:
        print("真实测试异常：" + type(error).__name__ + "；未输出服务详情。", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
