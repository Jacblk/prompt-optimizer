"""Offline dialogue handoff contracts; scripted reviews do not measure model quality."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
from types import SimpleNamespace
import unittest

from optimizer_config import ConfigurationError, ModelConfig
from optimizer_handoff import (
    DialogueHandoffPipeline, FidelityFailed, HistoryOptions,
    WindowExceeded, WindowLimits, handoff_conflict_questions, history_fingerprint,
    model_identity, prepare_history)
from optimizer_models import ModelReply


def models():
    return {role: ModelConfig(role, f"offline-{role}", "https://offline.invalid/v1", "placeholder", max_tokens=1024)
            for role in ("a", "b", "judge")}


def quote(doc, text=None):
    text = text or doc["text"]
    return {"chunk_id": doc["chunk_id"], "quote": text,
            "line_start": doc["line_start"] + doc["text"].count("\n", 0, doc["text"].index(text))}


def extracted(payload, *, conflict=False, current=False, details=False):
    doc = payload["stage_input"]["chunks"][0]
    citation = quote(doc)
    citations = [citation]
    if current:
        citations.append({"chunk_id": "current_request", "quote": payload["original_request"], "line_start": 1})
    return {"covered_chunk_ids": [doc["chunk_id"]], "items": [{
        "category": "constraint", "status": "conflict" if conflict else "confirmed",
        "text": "负责人 Alice 与 Bob 尚有冲突。" if conflict else "只读，不改文件。",
        "citations": citations, "source_item_ids": [],
        "details": [{"kind": "owner", "value": name, "citation": deepcopy(citation)} for name in ("Alice", "Bob")]
        if details else [],
    }], "reference_citations": []}


def merged(payload):
    data = payload["stage_input"]
    items, refs = [], []
    for summary in data["summaries"]:
        refs.extend(deepcopy(summary["reference_citations"]))
        for old in summary["items"]:
            item = deepcopy(old)
            item["source_item_ids"] = [item.pop("item_id")]
            items.append(item)
    return {"covered_chunk_ids": list(dict.fromkeys(c for s in data["summaries"] for c in s["covered_chunk_ids"])),
            "items": items, "reference_citations": refs}


def updated(payload):
    result = merged(payload)
    chunks = payload["stage_input"]["chunks"]
    result["covered_chunk_ids"] += [doc["chunk_id"] for doc in chunks]
    for item in result["items"]:
        answer = next((doc for doc in chunks if any(item_id in doc["related_item_ids"]
                                                   for item_id in item["source_item_ids"])), None)
        if item["status"] == "conflict" and answer:
            item["status"] = "superseded"
            item["text"] += " 已由明确回答解除。"
            item["citations"].append(quote(answer))
    for doc in chunks:
        citation = quote(doc)
        result["items"].append({"category": "constraint", "status": "confirmed", "text": doc["text"],
                                "citations": [citation], "source_item_ids": [],
                                "details": [{"kind": "owner", "value": "Alice", "citation": deepcopy(citation)}]
                                if "Alice" in doc["text"] else []})
    return result


def audited(payload):
    data = payload["stage_input"]
    chunks = [doc["chunk_id"] for doc in data["chunks"]]
    chunks += [c for summary in data["summaries"] for c in summary["covered_chunk_ids"]]
    return {"verdict": "pass", "checked_chunk_ids": list(dict.fromkeys(chunks)),
            "checked_item_ids": [item["item_id"] for s in data["summaries"] for item in s["items"]],
            "findings": [], "reason": "固定模拟检查。"}


class ScriptedCalls:
    def __init__(self, *, conflict=False, current=False, details=False, update=None, extractor=None):
        self.conflict, self.current, self.details = conflict, current, details
        self.update = update or updated
        self.extractor = extractor
        self.calls = []
        self.waiting = None

    async def __call__(self, role, system, payload, purpose):
        self.calls.append((role, system, deepcopy(payload), purpose))
        if self.waiting and payload["phase"] == "handoff_update":
            await self.waiting.wait()
        phase = payload["phase"]
        if phase == "handoff_extract":
            data = (self.extractor(payload) if self.extractor else
                    extracted(payload, conflict=self.conflict, current=self.current, details=self.details))
        elif phase == "handoff_merge":
            data = merged(payload)
        elif phase == "handoff_update":
            data = self.update(payload)
        else:
            data = audited(payload)
        return ModelReply(json.dumps(data, ensure_ascii=False), 7, 4, 11)


def pipeline(calls=None, *, history=None, window=256000, **options):
    calls = calls or ScriptedCalls()
    configs = models()
    windows = WindowLimits.from_config({"version": 1, "roles": {
        role: {**model_identity(config), "context_window": window} for role, config in configs.items()}}, configs)
    opts = SimpleNamespace(allow_repair=True, choose_layers=False, handoff_max_attempts=2,
                           max_input_chars=50000, **options)
    return DialogueHandoffPipeline(history or prepare_history(text="用户：只读，不改文件。"),
                                   windows, configs, opts, calls)


def resolution(state, text="负责人明确采用 Alice。"):
    question = handoff_conflict_questions(state)[0]
    return [{"question_id": question["question_id"], "text": text, "option_id": "alice",
             "related_item_ids": question["related_item_ids"]}]


class DialogueHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_versions_current_request_and_returns_independent_immutable_state(self):
        flow = pipeline(ScriptedCalls(current=True))
        state = await flow.full("继续只读。", revision=1)
        evidence = state.snapshot.payload()["evidence"]
        self.assertEqual(evidence[-1]["chunk_id"], "request-r1")
        self.assertEqual(evidence[-1]["source_sha256"], hashlib.sha256("继续只读。".encode()).hexdigest())
        state.evidence["request-r1"]["text"] = "污染"
        state.metadata()["stages"].clear()
        self.assertEqual(state.evidence["request-r1"]["text"], "继续只读。")
        self.assertTrue(state.metadata()["stages"])
        with self.assertRaises(FrozenInstanceError):
            state.revision = 9

    async def test_update_only_calls_update_and_preserves_previous_request_evidence(self):
        calls = ScriptedCalls(current=True)
        flow = pipeline(calls)
        state = await flow.full("继续只读。", revision=1)
        before = state.metadata()
        new = await flow.update(state, "继续只读，改用 Markdown。", [
            {"question_id": "format", "text": "改用 Markdown。", "option_id": None, "related_item_ids": []}], revision=2)
        self.assertEqual([call[2]["phase"] for call in calls.calls],
                         ["handoff_extract", "handoff_fidelity", "handoff_update", "handoff_fidelity"])
        self.assertEqual(state.metadata(), before)
        evidence = new.snapshot.payload()["evidence"]
        old = next(item for item in evidence if item["chunk_id"] == "request-r1")
        self.assertEqual(old["quote"], "继续只读。")
        self.assertEqual(old["source_sha256"], hashlib.sha256("继续只读。".encode()).hexdigest())
        self.assertEqual(new.snapshot.payload()["current_request_sha256"],
                         hashlib.sha256("继续只读，改用 Markdown。".encode()).hexdigest())
        self.assertIn("只读，不改文件。", new.snapshot.continuation_block)
        self.assertIn("Markdown", new.snapshot.continuation_block)
        with self.assertRaises(ConfigurationError):
            state.snapshot.attach("草稿", "继续只读，改用 Markdown。")

    async def test_format_correction_supersedes_only_format_and_retains_readonly(self):
        def separate_constraints(payload):
            doc = payload["stage_input"]["chunks"][0]
            return {"covered_chunk_ids": [doc["chunk_id"]], "items": [
                {"category": "constraint", "status": "confirmed", "text": text,
                 "citations": [quote(doc, text)], "source_item_ids": [], "details": []}
                for text in ("只读，不改文件", "输出 JSON")], "reference_citations": []}
        def replace_format(payload):
            data = updated(payload)
            old_format = next(item for item in data["items"] if item["text"] == "输出 JSON")
            old_format["status"] = "superseded"
            old_format["citations"].append(quote(payload["stage_input"]["chunks"][0]))
            return data
        flow = pipeline(ScriptedCalls(extractor=separate_constraints, update=replace_format),
                        history=prepare_history(text="用户：只读，不改文件；输出 JSON。"))
        state = await flow.full("继续。", revision=1)
        new = await flow.update(state, "继续只读，格式改为 Markdown。", [
            {"question_id": "format", "text": "格式改为 Markdown。", "related_item_ids": []}], revision=2)
        self.assertIn("只读，不改文件", new.snapshot.continuation_block)
        self.assertIn("Markdown", new.snapshot.continuation_block)
        self.assertNotIn("JSON", new.snapshot.continuation_block)
        archived = next(item for item in new.summary["items"] if item["status"] == "superseded")
        self.assertEqual(archived["citations"][0]["quote"], "输出 JSON")

    async def test_old_constraint_status_cannot_change_without_new_answer_evidence(self):
        def bad(payload):
            data = updated(payload)
            data["items"][0]["status"] = "superseded"
            return data
        flow = pipeline(ScriptedCalls(update=bad))
        state = await flow.full("继续。", revision=1)
        with self.assertRaises(FidelityFailed):
            await flow.update(state, "增加格式。", [{"question_id": "format", "text": "Markdown"}], revision=2)
        self.assertIn("只读", state.snapshot.continuation_block)

    async def test_unreviewed_snapshot_requires_a_new_full_preparation(self):
        from dataclasses import replace
        calls = ScriptedCalls()
        flow = pipeline(calls)
        state = await flow.full("继续。", revision=1)
        data = json.loads(state.snapshot.data_json)
        data["independently_reviewed"] = False
        state = replace(state, snapshot=replace(state.snapshot,
                        data_json=json.dumps(data, ensure_ascii=False)))
        flow._state = state
        with self.assertRaises(ConfigurationError):
            await flow.update(state, "增加格式。", [{"question_id": "format", "text": "Markdown"}], revision=2)
        self.assertEqual(len(calls.calls), 2)
        self.assertFalse(state.snapshot.payload()["independently_reviewed"])

    async def test_resolution_keeps_conflicting_details_archived_and_only_new_owner_active(self):
        calls = ScriptedCalls(conflict=True, details=True)
        flow = pipeline(calls, history=prepare_history(text="用户：负责人 Alice 或 Bob，未确认。"))
        state = await flow.full("继续确认负责人。", revision=1)
        question = handoff_conflict_questions(state)[0]
        self.assertEqual(question, handoff_conflict_questions(state)[0])
        new = await flow.update(state, "继续，负责人明确采用 Alice。", resolution(state), revision=2)
        self.assertFalse(handoff_conflict_questions(new))
        old = next(item for item in new.summary["items"] if item["status"] == "superseded")
        self.assertEqual({detail["value"] for detail in old["details"]}, {"Alice", "Bob"})
        active = [item for item in new.summary["items"] if item["status"] == "confirmed"]
        self.assertEqual([detail["value"] for item in active for detail in item["details"]], ["Alice"])
        self.assertNotIn("Bob", new.snapshot.continuation_block)
        self.assertEqual([call[0] for call in calls.calls], ["a", "judge", "a", "judge"])
        self.assertTrue(new.snapshot.payload()["independently_reviewed"])

    async def test_unrelated_answer_keeps_conflict_and_empty_answer_is_rejected(self):
        flow = pipeline(ScriptedCalls(conflict=True))
        state = await flow.full("继续。", revision=1)
        new = await flow.update(state, "继续，语气温和。", [
            {"question_id": "tone", "text": "语气温和。", "related_item_ids": []}], revision=2)
        self.assertTrue(handoff_conflict_questions(new))
        with self.assertRaises(ConfigurationError):
            await flow.update(new, "继续。", resolution(new, " "), revision=3)

    async def test_wrong_or_omitted_question_binding_cannot_resolve_conflict(self):
        calls = ScriptedCalls(conflict=True)
        flow = pipeline(calls)
        state = await flow.full("继续。", revision=1)
        for change in ({"question_id": "unrelated"}, {"related_item_ids": []}):
            answer = resolution(state)[0] | change
            with self.assertRaises(ConfigurationError):
                await flow.update(state, "采用 Alice。", [answer], revision=2)
        self.assertEqual(len(calls.calls), 2)
        self.assertTrue(handoff_conflict_questions(state))

    async def test_direct_conflict_upgrade_or_deletion_fails_even_with_fake_passing_judge(self):
        for kind in ("upgrade", "delete", "drop_details", "no_new_confirmed"):
            def bad(payload, kind=kind):
                data = updated(payload)
                if kind == "upgrade":
                    data["items"][0]["status"] = "confirmed"
                elif kind == "delete":
                    data["items"].pop(0)
                elif kind == "drop_details":
                    data["items"][0]["details"].pop()
                else:
                    data["items"].pop()
                return data
            calls = ScriptedCalls(conflict=True, details=True, update=bad)
            flow = pipeline(calls, history=prepare_history(text="用户：负责人 Alice 或 Bob，未确认。"))
            state = await flow.full("继续。", revision=1)
            old = state.snapshot.payload()
            with self.subTest(kind=kind), self.assertRaises(FidelityFailed):
                await flow.update(state, "采用 Alice。", resolution(state), revision=2)
            self.assertEqual(state.snapshot.payload(), old)
            self.assertIsNone(flow.metadata()["snapshot"])

    async def test_update_rejects_uncited_new_fact_and_unseen_cached_history_quote(self):
        for kind in ("new_fact", "unseen"):
            def bad(payload, kind=kind):
                data = updated(payload)
                old_citation = deepcopy(data["items"][0]["citations"][0])
                data["items"][-1]["citations"] = [old_citation]
                if kind == "unseen":
                    old_citation["quote"] = "用户："
                return data
            calls = ScriptedCalls(update=bad)
            flow = pipeline(calls)
            state = await flow.full("继续。", revision=1)
            with self.subTest(kind=kind), self.assertRaises(FidelityFailed):
                await flow.update(state, "采用新格式。", [{"question_id": "format", "text": "Markdown"}], revision=2)

    async def test_partial_answers_leave_other_conflicts_pending(self):
        history = prepare_history(text="用户：冲突一。\n用户：冲突二。", options=HistoryOptions(chunk_bytes=24))
        flow = pipeline(ScriptedCalls(conflict=True), history=history)
        state = await flow.full("继续。", revision=1)
        self.assertEqual(len(handoff_conflict_questions(state)), 2)
        new = await flow.update(state, "已确认第一项。", resolution(state, "第一项明确采用只读。"), revision=2)
        self.assertEqual(len(handoff_conflict_questions(new)), 1)

    async def test_full_and_updates_continue_beyond_prior_request_limits(self):
        calls = ScriptedCalls()
        flow = pipeline(calls)
        state = await flow.full('继续。', revision=1)
        for revision in range(2, 37):
            state = await flow.update(state, f'继续，第 {revision} 版采用 Markdown。',
                                      [{'question_id': 'format', 'text': 'Markdown'}], revision=revision)
        self.assertEqual(state.revision, 36)
        self.assertEqual(len(calls.calls), 72)
        self.assertEqual(flow.metadata()['status'], 'complete')
        self.assertEqual(flow.metadata()['preflight']['minimum_requests'], 5)
        self.assertNotIn('remaining_requests', flow.metadata()['preflight'])

    async def test_history_change_or_order_change_requires_full_and_stale_base_is_rejected(self):
        calls = ScriptedCalls()
        flow = pipeline(calls)
        state = await flow.full("继续。", revision=1)
        flow.history = prepare_history(text="用户：新目标。")
        with self.assertRaises(ConfigurationError):
            await flow.update(state, "继续。", [{"question_id": "detail", "text": "补充。"}], revision=2)
        new = await flow.full("分析新目标。", revision=2)
        self.assertNotEqual(new.history_fingerprint, state.history_fingerprint)
        with self.assertRaises(ConfigurationError):
            await flow.update(state, "继续。", [{"question_id": "detail", "text": "补充。"}], revision=3)
        self.assertEqual(len(calls.calls), 4)
        self.assertEqual(len(new.metadata()["previous_snapshots"]), 1)
        self.assertEqual(len(new.metadata()["stages"]), 2)

    async def test_cancel_update_preserves_previous_state_and_marks_incomplete_stage(self):
        calls = ScriptedCalls()
        flow = pipeline(calls)
        state = await flow.full("继续。", revision=1)
        calls.waiting = asyncio.Event()
        task = asyncio.create_task(flow.update(state, "继续，Markdown。", [{"question_id": "format", "text": "Markdown"}], revision=2))
        while not any(call[2]["phase"] == "handoff_update" for call in calls.calls):
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(flow.metadata()["status"], "cancelled")
        self.assertEqual(flow.metadata()["stages"][-1]["status"], "interrupted")
        self.assertEqual(state.metadata()["status"], "complete")
        self.assertIsNone(flow.metadata()["snapshot"])

    async def test_concurrent_update_and_nonincreasing_revision_are_rejected(self):
        calls = ScriptedCalls()
        flow = pipeline(calls)
        state = await flow.full("继续。", revision=1)
        with self.assertRaises(ConfigurationError):
            await flow.full("继续。", revision=1)
        calls.waiting = asyncio.Event()
        delta = [{"question_id": "format", "text": "Markdown"}]
        task = asyncio.create_task(flow.update(state, "继续，Markdown。", delta, revision=2))
        while not any(call[2]["phase"] == "handoff_update" for call in calls.calls):
            await asyncio.sleep(0)
        with self.assertRaises(ConfigurationError):
            await flow.update(state, "继续，Markdown。", delta, revision=3)
        calls.waiting.set()
        self.assertEqual((await task).revision, 2)

    async def test_window_overflow_does_not_publish_old_snapshot_or_call_update(self):
        calls = ScriptedCalls()
        flow = pipeline(calls, window=50000)
        state = await flow.full("继续。", revision=1)
        with self.assertRaises(WindowExceeded):
            await flow.update(state, "继续。", [{"question_id": "detail", "text": "中文" * 12000}], revision=2)
        self.assertEqual(len(calls.calls), 2)
        self.assertIsNone(flow.metadata()["snapshot"])
        self.assertEqual(state.metadata()["status"], "complete")


class HistoryIdentityTests(unittest.TestCase):
    def test_fingerprint_checks_source_order_and_cached_text_changes(self):
        first = prepare_history(text="用户：第一项。")
        second = prepare_history(text="用户：第二项。")
        first.sources.extend(second.sources)
        original = history_fingerprint(first)
        first.sources.reverse()
        self.assertNotEqual(history_fingerprint(first), original)
        first.sources.reverse()
        first.sources[0].document.page_content += "后续变化。"
        self.assertNotEqual(history_fingerprint(first), original)


if __name__ == "__main__":
    unittest.main()
