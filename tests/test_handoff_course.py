"""Course 12-14 continuation contracts; scripted answers are not AI quality scores."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from optimizer_config import ConfigurationError
from optimizer_handoff import prepare_history
from test_handoff import HistoryScripts, citation, extract, fidelity, merge, runner
from test_optimizer import draft, review


PLANNING = (
    "用户：只读分析，输出 JSON。\n"
    "用户：输出改为 Markdown，只读约束保持。\n"
    "Alice：当前还有两个功能未完成，集成测试尚未开始。\n"
    "Bob：建议将发布推迟至 12 月 1 日，需申请产品负责人的批准。\n"
    "会议记录：同意在下周二前安排完整测试计划评审，评审举行时间未定。\n"
    "用户：测试计划由 Carol 负责，相对日期基准和年份未给出。\n"
)


def planning_summary(payload):
    doc = payload["stage_input"]["chunks"][0]
    items = []

    def add(needle, category, status, text, *details):
        quote = next((line for line in doc["text"].splitlines() if needle in line), None)
        if quote is None:
            return
        source = citation(doc, quote)
        items.append({"category": category, "status": status, "text": text,
                      "citations": [source], "source_item_ids": [],
                      "details": [{"kind": kind, "value": value, "citation": deepcopy(source)}
                                  for kind, value in details]})

    add("输出 JSON", "constraint", "superseded", "旧 JSON 格式已被后续明确要求替代。")
    add("输出改为 Markdown", "constraint", "confirmed", "只读分析，输出 Markdown。")
    add("集成测试尚未开始", "progress", "unverified", "记录报告两个功能未完成、集成测试未开始。",
        ("blocker", "两个功能未完成"), ("dependency", "集成测试尚未开始"))
    add("需申请", "decision", "suggestion", "建议发布改为 12 月 1 日，尚待产品负责人批准。",
        ("proposed_time", "12 月 1 日"))
    add("下周二前", "next_step", "confirmed", "已同意在下周二前安排评审；尚未确认评审举行时间。",
        ("deadline", "下周二前"))
    add("Carol", "next_step", "confirmed", "测试计划由 Carol 负责；相对日期基准和年份待确认。",
        ("owner", "Carol"))
    return {"covered_chunk_ids": [doc["chunk_id"]], "items": items, "reference_citations": []}


def merge_details(payload):
    data = merge(payload)
    sources = {i["item_id"]: i for s in payload["stage_input"]["summaries"] for i in s["items"]}
    for item in data["items"]:
        item["details"] = []
        for item_id in item["source_item_ids"]:
            for detail in sources[item_id].get("details", []):
                if detail not in item["details"]:
                    item["details"].append(deepcopy(detail))
    return data


class ContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_reminder_preserves_active_constraints_and_excludes_old_format(self):
        scripts = HistoryScripts(handoff_extract=[planning_summary])
        result = await runner(scripts, history=prepare_history(text=PLANNING)).run("继续发布计划分析。")
        self.assertEqual(result.status, "ready", result.reason)
        snapshot = result.metadata["handoff"]["snapshot"]
        footer = snapshot["continuation_block"]
        self.assertTrue(result.optimized_prompt.endswith(footer))
        self.assertIn("只读分析，输出 Markdown", footer)
        self.assertNotIn("JSON", footer)
        self.assertIn("下周二前", snapshot["context_block"])
        self.assertIn("完成或安排工作的期限：下周二前", snapshot["context_block"])
        self.assertIn("原文明确的负责人：Carol", snapshot["context_block"])
        self.assertIn("提议的时间：12 月 1 日", snapshot["context_block"])
        self.assertIn("集成测试尚未开始", snapshot["context_block"])
        self.assertNotIn("2026", result.optimized_prompt)

    async def test_suggested_unverified_or_superseded_constraints_are_not_reaffirmed(self):
        text = "用户：只读。\n助手：建议部署。\n未知：允许删除待确认。\n用户：旧 JSON 已取消。"

        def states(payload):
            doc = payload["stage_input"]["chunks"][0]
            items = []
            for line, status in zip(doc["text"].splitlines(), ["confirmed", "suggestion", "unverified", "superseded"]):
                items.append({"category": "constraint", "status": status, "text": line,
                              "citations": [citation(doc, line)], "source_item_ids": []})
            return {"covered_chunk_ids": [doc["chunk_id"]], "items": items, "reference_citations": []}

        result = await runner(HistoryScripts(handoff_extract=[states]), history=prepare_history(text=text)).run("继续只读。")
        self.assertEqual(result.status, "ready")
        footer = result.metadata["handoff"]["snapshot"]["continuation_block"]
        self.assertIn("用户：只读。", footer)
        for old_line in text.splitlines()[1:]:
            self.assertNotIn(old_line, footer)
        self.assertTrue(result.metadata["handoff"]["snapshot"]["independently_reviewed"])
        self.assertNotIn("未经独立保真评审", footer)

    async def test_program_appends_reminder_after_model_instructions_and_repair(self):
        scripts = HistoryScripts(handoff_extract=[planning_summary], judge=[lambda p: review(p, "repair")])
        result = await runner(scripts, history=prepare_history(text=PLANNING)).run("继续分析，不改文件。")
        self.assertEqual(result.status, "ready")
        snapshot = result.metadata["handoff"]["snapshot"]
        for candidate in result.candidates:
            self.assertTrue(candidate["optimized_prompt"].endswith(snapshot["continuation_block"]))
        contexts = [p["handoff_context"] for _, _, p, _ in scripts.calls if "handoff_context" in p]
        self.assertTrue(any("repair" in p for _, _, p, _ in scripts.calls))
        self.assertEqual(len({c["snapshot_sha256"] for c in contexts}), 1)
        self.assertTrue(all(c["continuation_block"] == snapshot["continuation_block"] for c in contexts))

    async def test_finish_rejects_material_that_moves_reminder_away_from_end(self):
        engine = runner(HistoryScripts(handoff_extract=[planning_summary]), history=prepare_history(text=PLANNING))
        result = await engine.run("继续分析。")
        tampered = engine._finish("unreviewed", [], prompt=result.optimized_prompt + "\n额外要求：部署。")
        self.assertEqual(tampered.status, "needs_review")

    async def test_attach_keeps_reminder_at_end_when_model_copies_it_in_body(self):
        engine = runner()
        await engine.run("继续。")
        snapshot = engine.handoff_context
        prompt = snapshot.attach(snapshot.continuation_block + "\n后续内容。", "继续。")
        self.assertTrue(prompt.endswith(snapshot.continuation_block))

    async def test_tasks_with_same_history_receive_different_bound_snapshots(self):
        history = prepare_history(text="用户：只读分析。")
        first, second = runner(history=history), runner(history=history)
        results = await asyncio.gather(first.run("继续定位错误。"), second.run("只总结已知结论。"))
        snapshots = [r.metadata["handoff"]["snapshot"] for r in results]
        self.assertNotEqual(snapshots[0]["snapshot_sha256"], snapshots[1]["snapshot_sha256"])
        self.assertEqual(snapshots[0]["current_request_sha256"], hashlib.sha256("继续定位错误。".encode()).hexdigest())
        with self.assertRaisesRegex(ConfigurationError, "本轮任务不匹配"):
            first.handoff_context.attach("执行要求。", "只总结已知结论。")

    async def test_different_history_inputs_do_not_mix_between_parallel_tasks(self):
        results = await asyncio.gather(
            runner(history=prepare_history(text="用户：Alpha 仅分析订单。")).run("继续订单分析。"),
            runner(history=prepare_history(text="用户：Beta 仅分析退款。")).run("继续退款分析。"))
        self.assertEqual([r.status for r in results], ["ready", "ready"])
        self.assertIn("Alpha", results[0].optimized_prompt)
        self.assertNotIn("Beta", results[0].optimized_prompt)
        self.assertIn("Beta", results[1].optimized_prompt)
        self.assertNotIn("Alpha", results[1].optimized_prompt)


class DetailGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_optional_details_remains_compatible(self):
        result = await runner(HistoryScripts(handoff_extract=[extract])).run("继续。")
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.metadata["handoff"]["snapshot"]["summary"]["items"][0]["details"], [])

    async def test_fabricated_owner_date_or_dependency_cannot_use_unrelated_valid_quote(self):
        for kind, value in [("owner", "张三"), ("deadline", "2026-10-10"), ("dependency", "已安装数据库")]:
            def false_detail(payload):
                data = extract(payload)
                item = data["items"][0]
                item["details"] = [{"kind": kind, "value": value, "citation": deepcopy(item["citations"][0])}]
                return data

            with self.subTest(kind=kind):
                result = await runner(HistoryScripts(handoff_extract=[false_detail, false_detail])).run("继续。")
                self.assertEqual(result.status, "handoff_fidelity_failed")
                self.assertEqual(result.metadata["request_count"], 2)
                self.assertIsNone(result.optimized_prompt)

    async def test_detail_citation_must_be_in_the_same_item_evidence(self):
        def unrelated_detail(payload):
            data = planning_summary(payload)
            first, last = data["items"][0], data["items"][-1]
            first["details"] = deepcopy(last["details"])
            return data

        result = await runner(HistoryScripts(handoff_extract=[unrelated_detail, unrelated_detail]), history=prepare_history(text=PLANNING)).run("继续。")
        self.assertEqual(result.status, "handoff_fidelity_failed")

    async def test_identical_detail_is_not_repeated_in_one_item(self):
        def repeat_detail(payload):
            data = planning_summary(payload)
            item = next(i for i in data["items"] if i["details"])
            item["details"].append(deepcopy(item["details"][0]))
            return data

        result = await runner(HistoryScripts(handoff_extract=[repeat_detail, repeat_detail]), history=prepare_history(text=PLANNING)).run("继续。")
        self.assertEqual(result.status, "handoff_fidelity_failed")

    async def test_merge_cannot_drop_or_retag_source_time_relation(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / "一.md", Path(temp) / "二.md"]
            for path in paths:
                path.write_text(PLANNING, encoding="utf-8")
            history = prepare_history(paths)
            for retag in (False, True):
                def change_detail(payload):
                    data = merge_details(payload)
                    item = next(i for i in data["items"] if any(d["kind"] == "deadline" for d in i["details"]))
                    if retag:
                        for detail in item["details"]:
                            if detail["kind"] == "deadline":
                                detail["kind"] = "scheduled_time"
                    else:
                        item["details"] = []
                    return data

                scripts = HistoryScripts(handoff_extract=[planning_summary] * 2, handoff_merge=[change_detail] * 2)
                with self.subTest(retag=retag):
                    result = await runner(scripts, history=history).run("继续安排评审。")
                    self.assertEqual(result.status, "handoff_fidelity_failed")
                    self.assertIn("接续细节", result.metadata["handoff"]["stages"][-1]["attempts"][0]["reason"])

    async def test_merge_keeps_distinct_source_positions_and_details(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / "一.md", Path(temp) / "二.md"]
            for path in paths:
                path.write_text(PLANNING, encoding="utf-8")
            scripts = HistoryScripts(handoff_extract=[planning_summary] * 2, handoff_merge=[merge_details])
            result = await runner(scripts, history=prepare_history(paths)).run("继续规划。")
        self.assertEqual(result.status, "ready", result.reason)
        snapshot = result.metadata["handoff"]["snapshot"]
        owner = next(i for i in snapshot["summary"]["items"] if any(d["kind"] == "owner" for d in i["details"]))
        self.assertEqual(len(owner["details"]), 2)
        self.assertEqual({d["citation"]["chunk_id"] for d in owner["details"]},
                         {c["chunk_id"] for c in result.metadata["handoff"]["chunks"]})
        self.assertEqual({d["value"] for d in owner["details"]}, {"Carol"})

    async def test_judge_rejects_deadline_changed_to_meeting_time_despite_valid_quote(self):
        def bad_time(payload):
            data = planning_summary(payload)
            item = next(i for i in data["items"] if any(d["kind"] == "deadline" for d in i["details"]))
            item["text"] = "评审已确定于下周二举行。"
            item["details"][0]["kind"] = "scheduled_time"
            return data

        def reject(payload):
            self.assertIn("期限/举行时间/提议日期", scripts.calls[-1][1])
            return fidelity(payload, verdict="fail", kind="status_changed", input_quote="下周二前安排",
                            summary_quote="评审已确定于下周二举行。")

        scripts = HistoryScripts(handoff_extract=[bad_time] * 2, handoff_fidelity=[reject] * 2)
        result = await runner(scripts, history=prepare_history(text=PLANNING)).run("何时举行评审？")
        self.assertEqual(result.status, "handoff_fidelity_failed")
        self.assertIsNone(result.optimized_prompt)
        self.assertEqual(result.metadata["request_count"], 4)

    async def test_judge_rejects_speaker_promoted_to_owner_despite_name_in_quote(self):
        def bad_owner(payload):
            data = planning_summary(payload)
            item = next(i for i in data["items"] if any(d["kind"] == "proposed_time" for d in i["details"]))
            item["text"] = "Bob 已被指定负责发布。"
            item["details"].append({"kind": "owner", "value": "Bob", "citation": deepcopy(item["citations"][0])})
            return data

        scripts = HistoryScripts(handoff_extract=[bad_owner] * 2,
                                 handoff_fidelity=[lambda p: fidelity(p, verdict="fail", kind="fact_added",
                                     input_quote="Bob：建议", summary_quote="Bob 已被指定负责发布。")] * 2)
        result = await runner(scripts, history=prepare_history(text=PLANNING)).run("发布由谁负责？")
        self.assertEqual(result.status, "handoff_fidelity_failed")
        self.assertEqual(result.metadata["request_count"], 4)

    async def test_temporal_facts_reach_final_review_as_unchanged_reference_and_context(self):
        def check_final(payload):
            context = payload["handoff_context"]
            self.assertIn("完成或安排工作的期限：下周二前", context["context_block"])
            self.assertIn("评审举行时间未定", context["reference_blocks"][0])
            self.assertIn("12 月 1 日", context["context_block"])
            self.assertIn("尚待产品负责人批准", context["context_block"])
            for candidate in payload["candidates"]:
                self.assertTrue(candidate["text"].endswith(context["continuation_block"]))
            return review(payload)

        scripts = HistoryScripts(handoff_extract=[planning_summary], judge=[check_final])
        result = await runner(scripts, history=prepare_history(text=PLANNING)).run("继续安排评审，先确认未知时间。")
        self.assertEqual(result.status, "ready", result.reason)


if __name__ == "__main__":
    unittest.main()
