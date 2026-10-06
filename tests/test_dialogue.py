"""Dialogue contract tests with deterministic offline models; no configuration I/O."""
from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
import tempfile
import unittest

from optimizer_config import ConfigurationError, ModelConfig
from optimizer_dialogue import (
    ClarificationDecision, DialogueAnswer, DialogueController, DialogueQuestion,
    DialogueRequest, DialogueResponse)
from optimizer_engine import Optimizer, RunOptions
from optimizer_models import ModelCallError, ModelReply, OutputError, parse_output


ORIGINAL = "仅检查 src，不改文件。"


def question(identifier="model-question", text="需要检查哪些问题？", *, options=None):
    return {"id": identifier, "text": text, "reason": "答案会影响检查范围。",
            "options": options or [], "source": "clarification", "related_item_ids": []}


def sufficient(kind="detail"):
    return {"status": "sufficient", "questions": [], "change_kind": kind, "reason": "关键信息足够。"}


def ask(*questions, kind="detail"):
    return {"status": "ask", "questions": list(questions), "change_kind": kind, "reason": "需要确认范围。"}


def draft(text):
    return {"status": "ready", "optimized_prompt": text, "preserved_constraints": [],
            "clarification_questions": [], "change_summary": []}


def review(payload, action="select", *, questions=None):
    chosen = sorted(candidate["candidate_id"] for candidate in payload["candidates"]
                    if candidate["candidate_id"] != payload["original_candidate_id"])[0]
    if action == "keep_original":
        chosen = payload["original_candidate_id"]
    results = []
    for candidate in payload["candidates"]:
        failed = action == "repair" and candidate["candidate_id"] == chosen
        uncertain = action == "repair" and not failed
        findings = ([{"kind": "constraint_lost", "source_quote": "不改文件", "candidate_quote": "",
                      "explanation": "遗漏只读约束。"}] if failed else
                    [{"kind": "uncertain", "source_quote": payload["original_request"].strip()[:64],
                      "candidate_quote": "", "explanation": "离线修复场景未将此备选判为合格。"}] if uncertain else [])
        results.append({"candidate_id": candidate["candidate_id"],
            "verdict": "fail" if failed else "uncertain" if uncertain else "pass", "findings": findings,
            "clarity": 4, "conciseness": 4, "reason": "已核对完整已确认需求。"})
    return {"reviews": results, "action": action,
            "candidate_id": chosen if action in {"select", "keep_original", "repair"} else None,
            "clarification_questions": questions or ["检查范围还需要包含什么？"]
                if action == "needs_clarification" else [], "reason": "需要确认检查范围。"}


class DialogueScripts:
    def __init__(self, *, clarification=(), **queues):
        self.queues = {key: list(value) for key, value in queues.items()}
        self.queues["clarification"] = list(clarification)
        self.calls = []
        self.closed = []

    def factory(self, config):
        scripts = self

        class Model:
            def disable_streaming(self):
                self.streaming = False

            async def complete(self, system, payload, *, timeout, on_activity=None):
                scripts.calls.append((config.role, system, payload, timeout))
                is_clarification = payload.get("phase") == "dialogue_clarification"
                key = "clarification" if is_clarification else config.role
                queue = scripts.queues.get(key, [])
                value = queue.pop(0) if queue else (sufficient() if is_clarification
                    else review(payload) if config.role == "judge" else draft(payload["original_request"]))
                if isinstance(value, BaseException):
                    raise value
                if callable(value):
                    value = value(payload)
                if asyncio.iscoroutine(value):
                    value = await value
                if isinstance(value, ModelReply):
                    return value
                return ModelReply(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), 7, 4, 11)

            async def close(self):
                scripts.closed.append(config.role)

        return Model()


def make_controller(scripts=None, on_questions=None, on_event=None, **options):
    scripts = scripts or DialogueScripts()
    configs = {role: ModelConfig(role, "offline-" + role, "http://127.0.0.1:9/v1", "placeholder")
               for role in ("a", "b", "judge")}
    engine = Optimizer(configs, RunOptions(**options), factory=scripts.factory, rng=random.Random(42))
    return DialogueController(engine, on_questions=on_questions, on_event=on_event), scripts


class DialogueTests(unittest.IsolatedAsyncioTestCase):
    async def test_sufficient_has_one_clarification_then_generation_and_keeps_clients_until_close(self):
        controller, scripts = make_controller()
        result = await controller.submit(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.metadata["request_count"], 4)
        self.assertEqual(result.metadata["dialogue"]["original_request"], ORIGINAL)
        self.assertEqual(scripts.closed, [])
        await controller.close()
        await controller.close()
        self.assertCountEqual(scripts.closed, ["a", "b", "judge"])

    async def test_continuous_clarification_preserves_raw_answers_and_does_not_adopt_other_options(self):
        scripts = DialogueScripts(clarification=[ask(question(options=[
            {"id": "recommended", "label": "仅检查异常处理"},
            {"id": "other", "label": "同时修改所有业务代码"}])), ask(question(text="验收方式是什么？")), sufficient()])
        batches = []

        async def answer(batch):
            batches.append(batch)
            if len(batches) == 1:
                return {"answers": [{"question_id": batch.questions[0].id, "option_id": "recommended"}]}
            return DialogueResponse(answers=[DialogueAnswer(question_id=batch.questions[0].id, text="输出风险列表，保持只读。")])

        controller, _ = make_controller(scripts, answer)
        result = await controller.submit(ORIGINAL)
        data = result.metadata["dialogue"]
        self.assertEqual(data["total_rounds"], 2)
        self.assertEqual(data["updates"][0]["raw_text"], "")
        self.assertIn("仅检查异常处理", data["confirmed_request"])
        self.assertNotIn("同时修改所有业务代码", data["confirmed_request"])
        self.assertEqual(data["pending_questions"], [])
        await controller.close()

    async def test_partial_answers_keep_unanswered_question_id(self):
        scripts = DialogueScripts(clarification=[ask(question("first"), question("second", "验收方式？")),
            lambda payload: ask(payload["pending_questions"][0]), sufficient()])
        seen = []

        async def answer(batch):
            seen.append([question.id for question in batch.questions])
            return {"answers": [{"question_id": batch.questions[0].id, "text": "仅检查异常处理"}]}

        controller, _ = make_controller(scripts, answer)
        result = await controller.submit(ORIGINAL)
        self.assertEqual(seen[1], [seen[0][1]])
        self.assertEqual(len(result.metadata["dialogue"]["updates"]), 2)
        await controller.close()

    async def test_model_can_ask_four_to_seven_rounds_without_manual_resume(self):
        for rounds in (4, 7):
            with self.subTest(rounds=rounds):
                scripts = DialogueScripts(clarification=[
                    ask(question(text=f"确认范围 {index}？")) for index in range(1, rounds + 1)
                ] + [sufficient()])
                seen = []

                async def answer(batch):
                    seen.append(batch.round_number)
                    self.assertNotIn("round_limit", batch.to_dict())
                    return {"answers": [{"question_id": batch.questions[0].id,
                                         "text": f"范围回答 {len(seen)}"}]}

                controller, _ = make_controller(scripts, answer)
                result = await controller.submit(ORIGINAL)
                self.assertEqual(result.status, "ready")
                self.assertEqual(result.metadata["request_count"], rounds + 4)
                self.assertEqual(seen, list(range(1, rounds + 1)))
                self.assertEqual(result.metadata["dialogue"]["total_rounds"], rounds)
                self.assertNotIn("allowed_rounds", result.metadata["dialogue"])
                self.assertEqual(result.metadata["dialogue"]["status"], "open")
                self.assertEqual(scripts.closed, [])
                await controller.close()

    async def test_pause_resume_and_legacy_request_reopen_questions_without_model_call(self):
        for entry in ("resume", "legacy_request"):
            with self.subTest(entry=entry):
                scripts = DialogueScripts(clarification=[ask(question()), sufficient()])
                batches = []

                async def answer(batch):
                    batches.append(batch)
                    if len(batches) == 1:
                        return {"action": "pause"}
                    self.assertEqual(len(scripts.calls), 1)
                    self.assertEqual(batch.to_dict(), batches[0].to_dict())
                    return {"answers": [{"question_id": batch.questions[0].id, "text": "仅异常处理"}]}

                controller, _ = make_controller(scripts, answer)
                paused = await controller.submit(ORIGINAL)
                self.assertEqual(paused.metadata["dialogue"]["revision"], 1)
                self.assertEqual(paused.metadata["dialogue"]["total_rounds"], 0)
                if entry == "resume":
                    result = await controller.resume()
                else:
                    result = await controller.optimizer.run_dialogue(
                        DialogueRequest(ORIGINAL, continue_rounds=True), controller._questions)
                self.assertEqual(result.status, "ready")
                self.assertEqual(result.metadata["request_count"], 5)
                self.assertEqual(result.metadata["dialogue"]["session_id"], paused.metadata["dialogue"]["session_id"])
                self.assertEqual(len(result.metadata["dialogue"]["updates"]), 1)
                self.assertEqual(paused.metadata["dialogue"]["revision"], 1)
                await controller.close()

    async def test_resume_requires_a_paused_session(self):
        controller, scripts = make_controller()
        with self.assertRaises(ConfigurationError):
            await controller.resume()
        await controller.submit(ORIGINAL)
        with self.assertRaises(ConfigurationError):
            await controller.resume()
        self.assertEqual(len(scripts.calls), 4)
        await controller.close()

    async def test_resume_with_changed_material_rechecks_before_showing_questions(self):
        from optimizer_documents import ReferenceFile, prepare_references
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "material.md"
            path.write_text("新材料：验收为中文问题列表。", encoding="utf-8")
            references = prepare_references([ReferenceFile(path)], query=ORIGINAL)
            scripts = DialogueScripts(clarification=[ask(question(text="旧材料的问题？")),
                ask(question(text="新材料的验收范围？")), sufficient()])
            batches = []

            async def answer(batch):
                batches.append(batch)
                if len(batches) == 1:
                    return {"action": "pause"}
                self.assertEqual(len(scripts.calls), 2)
                self.assertEqual(batch.questions[0].text, "新材料的验收范围？")
                self.assertNotEqual(batch.questions[0].id, batches[0].questions[0].id)
                return {"answers": [{"question_id": batch.questions[0].id, "text": "输出中文问题列表"}]}

            controller, _ = make_controller(scripts, answer)
            paused = await controller.submit(ORIGINAL)
            result = await controller.optimizer.run_dialogue(
                DialogueRequest(ORIGINAL, resume=True, references=references), controller._questions)
            self.assertEqual(result.status, "ready")
            self.assertEqual(result.metadata["request_count"], 6)
            self.assertNotEqual(result.metadata["dialogue"]["fingerprint"], paused.metadata["dialogue"]["fingerprint"])
            self.assertIn("新材料", str(scripts.calls[1][2]))
            self.assertEqual(len(result.metadata["dialogue"]["updates"]), 1)
            await controller.close()

    async def test_continuous_questions_can_exceed_prior_request_limits(self):
        scripts = DialogueScripts(clarification=[ask(question(text=f"关键项 {n}？")) for n in range(70)]
                                  + [sufficient()])

        async def answer(batch):
            return {"answers": [{"question_id": batch.questions[0].id, "text": "已明确确认"}]}

        controller, _ = make_controller(scripts, answer)
        result = await controller.submit(ORIGINAL)
        self.assertEqual(result.status, "ready", result.reason)
        self.assertEqual(result.metadata["request_count"], 74)
        self.assertEqual(result.metadata["dialogue"]["total_rounds"], 70)
        self.assertFalse(result.metadata["dialogue"]["closed"])
        await controller.close()

    async def test_third_answer_sufficient_generates_without_a_pause(self):
        scripts = DialogueScripts(clarification=[ask(question(text=f"确认关键项 {index}？")) for index in range(3)])
        scripts.queues["clarification"].append(sufficient())

        async def answer(batch):
            return {"answers": [{"question_id": batch.questions[0].id, "text": "已明确确认。"}]}

        controller, _ = make_controller(scripts, answer)
        result = await controller.submit(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.metadata["dialogue"]["total_rounds"], 3)
        self.assertEqual(result.metadata["dialogue"]["status"], "open")
        await controller.close()

    async def test_user_wait_and_idle_are_excluded_from_elapsed_statistics(self):
        scripts = DialogueScripts(clarification=[ask(question()), sufficient()])

        async def answer(batch):
            await asyncio.sleep(0.25)
            return {"answers": [{"question_id": batch.questions[0].id, "text": "异常处理"}]}

        controller, _ = make_controller(scripts, answer)
        first = await controller.submit(ORIGINAL)
        self.assertEqual(first.status, "ready")
        self.assertGreaterEqual(first.metadata["user_wait_seconds"], 0.2)
        await asyncio.sleep(0.25)
        frozen = controller.optimizer.metadata()["elapsed_seconds"]
        second = await controller.submit("再补充：用 Markdown 表格。")
        self.assertEqual(second.status, "ready")
        self.assertLess(frozen, 0.2)
        self.assertLess(second.metadata["elapsed_seconds"], 0.2)
        self.assertEqual(second.metadata["request_count"], 9)
        await controller.close()

    async def test_metadata_queries_during_question_wait_freeze_elapsed(self):
        controller = None

        async def answer(batch):
            before = controller.optimizer.metadata()["elapsed_seconds"]
            await asyncio.sleep(0.06)
            after = controller.optimizer.metadata()["elapsed_seconds"]
            self.assertEqual(before, after)
            return {"answers": [{"question_id": batch.questions[0].id, "text": "确认检查范围。"}]}

        controller, _ = make_controller(DialogueScripts(clarification=[ask(question()), sufficient()]), answer)
        result = await controller.submit(ORIGINAL)
        self.assertEqual(result.status, "ready")
        await controller.close()

    async def test_c_question_returns_to_same_dialogue_and_original_candidate_uses_full_confirmation(self):
        scripts = DialogueScripts(clarification=[ask(question(text=f"确认关键项 {n}？")) for n in range(3)] + [sufficient()],
            judge=[lambda payload: review(payload, "needs_clarification"),
                                         lambda payload: review(payload, "keep_original")])

        async def answer(batch):
            self.assertEqual(batch.questions[0].source, "judge" if batch.round_number == 4 else "clarification")
            return {"answers": [{"question_id": batch.questions[0].id, "text": "仅异常处理，不执行修改。"}]}

        controller, _ = make_controller(scripts, answer)
        result = await controller.submit(ORIGINAL)
        self.assertEqual(result.status, "ready")
        confirmed = result.metadata["dialogue"]["confirmed_request"]
        self.assertEqual(result.optimized_prompt, confirmed)
        latest_review = [payload for role, _, payload, _ in scripts.calls if role == "judge"][-1]
        self.assertEqual(latest_review["original_request"], confirmed)
        original = next(candidate for candidate in latest_review["candidates"]
                        if candidate["candidate_id"] == latest_review["original_candidate_id"])
        self.assertEqual(original["text"], confirmed)
        self.assertEqual(result.metadata["request_count"], 11)
        self.assertEqual(result.metadata["dialogue"]["total_rounds"], 4)
        await controller.close()

    async def test_pending_generator_is_checked_by_judge_after_three_clarification_rounds(self):
        scripts = DialogueScripts(
            clarification=[ask(question(text=f"确认关键项 {n}？")) for n in range(3)] + [sufficient()],
            a=[lambda payload: {**draft(payload["original_request"]), "status": "needs_clarification",
                                    "clarification_questions": ["还要检查缓存处理吗？"]}],
            judge=[lambda payload: review(payload, "needs_clarification")])

        async def answer(batch):
            self.assertEqual(batch.questions[0].source,
                             "judge" if batch.round_number == 4 else "clarification")
            return {"answers": [{"question_id": batch.questions[0].id, "text": "检查缓存处理，保持只读。"}]}

        controller, _ = make_controller(scripts, answer)
        result = await controller.submit(ORIGINAL)
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.metadata["dialogue"]["total_rounds"], 4)
        self.assertEqual(result.metadata["request_count"], 11)
        self.assertIn("检查缓存处理", result.optimized_prompt)
        await controller.close()

    async def test_prompt_repair_once_across_c_return_and_future_submissions(self):
        scripts = DialogueScripts(judge=[lambda payload: review(payload, "repair"),
            lambda payload: review(payload, "needs_clarification"), lambda payload: review(payload, "repair"),
            lambda payload: review(payload, "repair")])

        async def answer(batch):
            return {"answers": [{"question_id": batch.questions[0].id, "text": "异常处理"}]}

        controller, _ = make_controller(scripts, answer)
        first = await controller.submit(ORIGINAL)
        self.assertEqual(first.status, "needs_review")
        second = await controller.submit("再确认：只检查，不修改。")
        self.assertEqual(second.status, "needs_review")
        self.assertEqual(second.metadata["dialogue"]["prompt_repairs_used"], 1)
        self.assertEqual(sum("repair" in payload for _, _, payload, _ in scripts.calls), 1)
        await controller.close()

    async def test_repeated_submissions_exceed_prior_limits_in_the_same_session(self):
        controller, scripts = make_controller()
        session_id = None
        for revision in range(1, 19):
            result = await controller.submit(ORIGINAL if revision == 1 else f"补充 {revision}：输出列表。")
            self.assertEqual(result.status, "ready", result.reason)
            session_id = session_id or result.metadata["dialogue"]["session_id"]
            self.assertEqual(result.metadata["dialogue"]["session_id"], session_id)
            self.assertEqual(result.metadata["request_count"], revision * 4)
            self.assertEqual(result.metadata["dialogue"]["revision"], revision)
            self.assertFalse(result.metadata["dialogue"]["closed"])
        self.assertEqual(scripts.closed, [])
        await controller.close()
        self.assertCountEqual(scripts.closed, ["a", "b", "judge"])

    async def test_known_and_unknown_token_budgets(self):
        for scripts, budget in ((DialogueScripts(), 11),
                                (DialogueScripts(clarification=[ModelReply(json.dumps(sufficient()))]), 100)):
            controller, _ = make_controller(scripts, token_budget=budget)
            result = await controller.submit(ORIGINAL)
            self.assertEqual(result.status, "budget_exceeded")
            self.assertEqual(len(scripts.calls), 1)
            self.assertTrue(result.metadata["dialogue"]["closed"])

    async def test_cancel_model_and_waiting_question_close_and_keep_checkpoint(self):
        for phase in ("model", "question"):
            started = asyncio.Event()

            async def slow(payload):
                started.set()
                await asyncio.Event().wait()

            async def answer(batch):
                started.set()
                await asyncio.Event().wait()

            scripts = DialogueScripts(clarification=[slow if phase == "model" else ask(question())])
            events = []
            controller, _ = make_controller(scripts, answer, events.append)
            task = asyncio.create_task(controller.submit(ORIGINAL))
            await asyncio.wait_for(started.wait(), 1)
            result = await controller.cancel()
            self.assertEqual((await task).status, "cancelled")
            self.assertEqual(result.status, "cancelled")
            self.assertEqual(scripts.closed, ["a"])
            self.assertEqual(result.metadata["dialogue"]["original_request"], ORIGINAL)
            if phase == "question":
                self.assertTrue(result.metadata["dialogue"]["pending_questions"])
                self.assertTrue(any(event["kind"] == "questions" for event in events))

    async def test_cancel_idle_and_callback_error_are_terminal(self):
        controller, scripts = make_controller()
        await controller.submit(ORIGINAL)
        cancelled = await controller.cancel()
        self.assertEqual(cancelled.status, "cancelled")
        self.assertCountEqual(scripts.closed, ["a", "b", "judge"])
        self.assertIsNone(await controller.cancel())

        def failed_callback(batch):
            raise RuntimeError("sensitive-provider-detail")

        controller, scripts = make_controller(DialogueScripts(clarification=[ask(question())]), failed_callback)
        failed = await controller.submit(ORIGINAL)
        self.assertEqual(failed.status, "failed")
        self.assertNotIn("sensitive-provider-detail", failed.reason)
        self.assertEqual(scripts.closed, ["a"])

    async def test_duplicate_submit_and_api_mixing_are_rejected_without_spending_requests(self):
        started, finish = asyncio.Event(), asyncio.Event()

        async def slow(payload):
            started.set()
            await finish.wait()
            return sufficient()

        controller, scripts = make_controller(DialogueScripts(clarification=[slow]))
        task = asyncio.create_task(controller.submit(ORIGINAL))
        await started.wait()
        with self.assertRaises(ConfigurationError):
            await controller.submit("第二个提交")
        with self.assertRaises(ConfigurationError):
            await controller.optimizer.run(ORIGINAL)
        finish.set()
        await task
        self.assertEqual(len(scripts.calls), 4)
        await controller.close()
        other, _ = make_controller()
        await other.optimizer.run(ORIGINAL)
        with self.assertRaises(ConfigurationError):
            await other.submit(ORIGINAL)

    async def test_correction_invalidates_old_result_and_snapshots_are_detached(self):
        controller, _ = make_controller()
        first = await controller.submit("仅检查，不改文件；输出 JSON。")
        self.assertTrue(controller.is_current(first))
        second = await controller.submit("纠正输出要求：改为 Markdown，其余只读约束保持。")
        self.assertFalse(controller.is_current(first))
        self.assertTrue(controller.is_current(second))
        state = second.metadata["dialogue"]
        self.assertIn("改为 Markdown", state["confirmed_request"])
        self.assertEqual(first.metadata["dialogue"]["updates"], [])
        self.assertGreater(state["revision"], first.metadata["dialogue"]["revision"])
        candidate = second.candidates[0]
        self.assertEqual(candidate["fingerprint"], state["fingerprint"])
        controller.snapshot()["updates"][0]["text"] = "external mutation"
        self.assertNotIn("external mutation", controller.snapshot()["confirmed_request"])
        for archive in state["archived_results"]:
            self.assertNotIn("dialogue", archive["result"]["metadata"])
        await controller.close()

    async def test_invalid_combined_submission_keeps_existing_session_usable(self):
        controller, scripts = make_controller(max_input_chars=40)
        first = await controller.submit("检查。")
        with self.assertRaises(ConfigurationError):
            await controller.submit("新补充" * 8)
        self.assertTrue(controller.is_current(first))
        self.assertEqual(controller.snapshot()["revision"], 1)
        self.assertEqual(len(scripts.calls), 4)
        self.assertFalse(controller.snapshot()["closed"])
        await controller.close()

    async def test_oversized_answer_keeps_revision_pending_and_budget_for_correction(self):
        responses = 0

        async def answer(batch):
            nonlocal responses
            responses += 1
            return {"answers": [{"question_id": batch.questions[0].id,
                                 "text": "大" * 350 if responses == 1 else "异常处理"}]}

        controller, scripts = make_controller(DialogueScripts(clarification=[ask(question()), sufficient()]),
                                             answer, max_input_chars=260)
        paused = await controller.submit(ORIGINAL)
        info = paused.metadata["dialogue"]
        self.assertEqual(info["status"], "paused")
        self.assertEqual(info["revision"], 1)
        self.assertEqual(info["updates"], [])
        self.assertEqual(info["total_rounds"], 0)
        self.assertTrue(info["pending_questions"])
        self.assertFalse(info["closed"])
        result = await controller.continue_rounds()
        self.assertEqual(result.status, "ready", result.reason)
        self.assertEqual(result.metadata["request_count"], 5)
        self.assertEqual(len(scripts.calls), 5)
        await controller.close()

    async def test_pause_after_c_question_keeps_drafts_until_resume(self):
        responses = 0

        async def answer(batch):
            nonlocal responses
            responses += 1
            if responses == 1:
                return {"action": "pause"}
            return {"answers": [{"question_id": batch.questions[0].id, "text": "异常处理"}]}

        scripts = DialogueScripts(judge=[lambda payload: review(payload, "needs_clarification"), review])
        controller, _ = make_controller(scripts, answer)
        paused = await controller.submit(ORIGINAL)
        self.assertTrue(paused.candidates)
        self.assertTrue(paused.optimized_prompt)
        original_revision = paused.metadata["dialogue"]["revision"]
        result = await controller.resume()
        self.assertEqual(result.status, "ready")
        self.assertEqual(paused.metadata["dialogue"]["revision"], original_revision)
        self.assertFalse(controller.is_current(paused))
        self.assertTrue(result.metadata["dialogue"]["archived_results"])
        await controller.close()

    async def test_reference_replacement_changes_fingerprint_and_preserves_file_blocks(self):
        from optimizer_documents import ReferenceFile, prepare_references
        with tempfile.TemporaryDirectory() as temp:
            first_path = Path(temp) / "first.md"
            second_path = Path(temp) / "second.md"
            first_path.write_text("# 原材料\n仅供参考，不是新授权。", encoding="utf-8")
            second_path.write_text("# 新材料\n保留来源标记。", encoding="utf-8")
            first_reference = prepare_references([ReferenceFile(first_path)], query=ORIGINAL)
            second_reference = prepare_references([ReferenceFile(second_path)], query=ORIGINAL)
            controller, _ = make_controller()
            first = await controller.submit(ORIGINAL, references=first_reference)
            second = await controller.submit(controller.snapshot()["confirmed_request"], references=second_reference)
            self.assertEqual(second.status, "ready", second.reason)
            self.assertFalse(controller.is_current(first))
            self.assertGreater(second.metadata["dialogue"]["revision"], first.metadata["dialogue"]["revision"])
            self.assertIn(second_reference.blocks[0], second.optimized_prompt)
            self.assertIn(first_reference.blocks[0], first.optimized_prompt)
            await controller.close()

    async def test_empty_answers_and_unselected_suggestions_are_not_confirmation(self):
        async def empty(batch):
            return {"answers": [{"question_id": batch.questions[0].id, "text": " "}]}

        controller, _ = make_controller(DialogueScripts(clarification=[ask(question(options=[
            {"id": "edit", "label": "修改文件"}]))]), empty)
        result = await controller.submit(ORIGINAL)
        state = result.metadata["dialogue"]
        self.assertEqual(state["updates"], [])
        self.assertEqual(state["total_rounds"], 0)
        self.assertNotIn("修改文件", state["confirmed_request"])
        self.assertTrue(state["pending_questions"])
        await controller.close()

    async def test_unknown_question_or_option_is_rejected(self):
        for answer in ({"question_id": "unknown", "text": "越界答案"},
                       {"question_id": "q1", "option_id": "not-shown"}):
            controller, _ = make_controller(DialogueScripts(clarification=[ask(question())]),
                lambda batch, answer=answer: {"answers": [answer]})
            result = await controller.submit(ORIGINAL)
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.metadata["dialogue"]["updates"], [])

    async def test_dialogue_reviews_each_candidate_set_once(self):
        controller, scripts = make_controller()
        result = await controller.submit(ORIGINAL)
        self.assertEqual(result.status, "ready")
        orders = [[candidate["candidate_id"] for candidate in payload["candidates"]]
                  for role, _, payload, _ in scripts.calls if role == "judge"]
        self.assertEqual(len(orders), 1)
        self.assertEqual(result.metadata["request_count"], 4)
        await controller.close()

    async def test_network_timeout_and_model_failure_preserve_session_report(self):
        from dataclasses import replace
        async def slow(payload):
            raise TimeoutError("offline network timeout")

        for output in (slow, ModelCallError("offline failure")):
            controller, scripts = make_controller(DialogueScripts(clarification=[output]), retries=0)
            controller.optimizer.configs["a"] = replace(controller.optimizer.configs["a"], timeout=0.02)
            result = await controller.submit(ORIGINAL)
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.metadata["dialogue"]["original_request"], ORIGINAL)
            self.assertEqual(scripts.closed, ["a"])

    async def test_normal_pause_resume_preserves_confirmed_update_delta(self):
        shown = 0
        scripts = DialogueScripts(clarification=[ask(question()),
            ask(question("format", "输出什么格式？")), sufficient()])

        async def answer(batch):
            nonlocal shown
            shown += 1
            if shown == 2:
                return {"action": "pause"}
            return {"answers": [{"question_id": batch.questions[0].id,
                                  "text": "异常处理" if shown == 1 else "Markdown"}]}

        controller, _ = make_controller(scripts, answer)
        paused = await controller.submit(ORIGINAL)
        self.assertEqual(paused.metadata["dialogue"]["status"], "paused")
        result = await controller.resume()
        self.assertEqual(result.status, "ready", result.reason)
        clarification = [payload for _, _, payload, _ in scripts.calls
                         if payload.get("phase") == "dialogue_clarification"][-1]
        self.assertEqual([item["text"] for item in clarification["latest_updates"]], ["异常处理", "Markdown"])
        await controller.close()


class DialogueSchemaTests(unittest.TestCase):
    def test_schema_rejects_questions_when_information_is_sufficient(self):
        with self.assertRaises(OutputError):
            parse_output(json.dumps({**sufficient(), "questions": [question()]}), ClarificationDecision)

    def test_schema_limits_questions_and_rejects_duplicate_options(self):
        with self.assertRaises(OutputError):
            parse_output(json.dumps(ask(*(question(str(index)) for index in range(4)))), ClarificationDecision)
        with self.assertRaises(ValueError):
            DialogueQuestion.model_validate(question(options=[{"id": "same", "label": "A"},
                                                              {"id": "same", "label": "B"}]))


class DialogueHandoffIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def make_handoff(self, *, conflict=False, clarification=(), on_questions=None, **options):
        from test_dialogue_handoff import ScriptedCalls
        from optimizer_handoff import model_identity, prepare_history
        history_calls = ScriptedCalls(conflict=conflict, current=True, details=conflict)

        class IntegratedScripts(DialogueScripts):
            def factory(self, config):
                base = super().factory(config)
                scripts = self

                class Model:
                    def disable_streaming(self):
                        base.disable_streaming()

                    async def complete(self, system, payload, *, timeout, on_activity=None):
                        if payload.get("phase", "").startswith("handoff_"):
                            scripts.calls.append((config.role, system, payload, timeout))
                            return await history_calls(config.role, system, payload, "offline")
                        return await base.complete(system, payload, timeout=timeout, on_activity=on_activity)

                    async def close(self):
                        await base.close()

                return Model()

        scripts = IntegratedScripts(clarification=clarification)
        controller, _ = make_controller(scripts, on_questions, **options)
        engine = controller.optimizer
        engine.history = prepare_history(text="用户：仅检查，不改文件。负责人 Alice 与 Bob 未定。")
        engine.context_windows = {"version": 1, "roles": {
            role: {**model_identity(config), "context_window": 256000} for role, config in engine.configs.items()}}
        return controller, scripts, history_calls

    async def test_workflow_switches_preserve_session_usage_and_retired_reports(self):
        controller, scripts, history_calls = self.make_handoff()
        engine = controller.optimizer
        history, windows = engine.history, engine.context_windows
        engine.history = engine.context_windows = None
        first = await controller.submit(ORIGINAL)
        self.assertEqual(first.metadata["workflow"], "normal")
        self.assertNotIn("max_requests", first.metadata["limits"])
        self.assertNotIn("total_timeout", first.metadata["limits"])
        second = await controller.submit(controller.snapshot()["confirmed_request"],
                                         history=history, context_windows=windows)
        self.assertEqual(second.status, "ready", second.reason)
        self.assertEqual(second.metadata["workflow"], "handoff")
        self.assertNotIn("max_requests", second.metadata["limits"])
        self.assertNotIn("total_timeout", second.metadata["limits"])
        old_context = engine.handoff_context
        third = await controller.submit(controller.snapshot()["confirmed_request"],
                                        history=None, context_windows=None)
        self.assertEqual(third.status, "ready", third.reason)
        self.assertEqual(third.metadata["workflow"], "normal")
        self.assertNotIn("handoff", third.metadata)
        self.assertNotIn(old_context.context_block, third.optimized_prompt)
        for field in ("handoff_context", "history_pipeline", "windows", "_dialogue_handoff_state",
                      "_dialogue_handoff_identity", "_dialogue_last_handoff_pipeline", "context_windows"):
            self.assertIsNone(getattr(engine, field), field)
        fourth = await controller.submit(controller.snapshot()["confirmed_request"],
                                         history=history, context_windows=windows)
        self.assertEqual(fourth.status, "ready", fourth.reason)
        self.assertEqual([call[2]["phase"] for call in history_calls.calls].count("handoff_extract"), 2)
        for result in (first, second, third, fourth):
            self.assertEqual(result.metadata["dialogue"]["session_id"], first.metadata["dialogue"]["session_id"])
            self.assertEqual(result.metadata["dialogue"]["updates"], [])
        self.assertEqual([result.metadata["dialogue"]["revision"] for result in (first, second, third, fourth)], [1, 2, 3, 4])
        self.assertEqual([result.metadata["request_count"] for result in (first, second, third, fourth)], [4, 10, 14, 20])
        self.assertEqual(fourth.metadata["known_total_tokens"], 220)
        self.assertGreaterEqual(fourth.metadata["elapsed_seconds"], first.metadata["elapsed_seconds"])
        self.assertNotIn("max_requests", third.metadata["limits"])
        self.assertNotIn("total_timeout", third.metadata["limits"])
        archived = next(item for item in fourth.metadata["dialogue"]["archived_results"]
                        if item["revision"] == second.metadata["dialogue"]["revision"])
        self.assertEqual(archived["result"]["metadata"]["handoff"]["snapshot"], second.metadata["handoff"]["snapshot"])
        self.assertEqual(scripts.closed, [])
        await controller.close()

    async def test_switch_does_not_reset_token_budget(self):
        controller, _, _ = self.make_handoff(token_budget=10000)
        engine = controller.optimizer
        history, windows = engine.history, engine.context_windows
        engine.history = engine.context_windows = None
        await controller.submit(ORIGINAL)
        handoff = await controller.submit(controller.snapshot()["confirmed_request"],
                                         history=history, context_windows=windows)
        normal = await controller.submit(controller.snapshot()["confirmed_request"], history=None)
        for result in (handoff, normal):
            self.assertEqual(result.status, "ready", result.reason)
            self.assertEqual(result.metadata["limits"]["token_budget"], 10000)
        self.assertEqual(normal.metadata["known_total_tokens"], 154)
        await controller.close()

    async def test_invalid_history_or_window_keeps_existing_session_usable(self):
        controller, scripts, _ = self.make_handoff()
        engine = controller.optimizer
        history, windows = engine.history, engine.context_windows
        engine.history = engine.context_windows = None
        first = await controller.submit(ORIGINAL)
        for new_history, new_windows in (("invalid bundle", windows), (history, None),
                                         (history, {"version": 1, "roles": {}})):
            with self.subTest(history=type(new_history).__name__), self.assertRaises(ConfigurationError):
                await controller.submit(controller.snapshot()["confirmed_request"],
                                        history=new_history, context_windows=new_windows)
            self.assertTrue(controller.is_current(first))
            self.assertEqual(len(scripts.calls), 4)
            self.assertEqual(controller.snapshot()["revision"], 1)
            self.assertFalse(controller.snapshot()["closed"])
            self.assertNotIn("max_requests", engine.metadata()["limits"])
        await controller.close()

    async def test_clearing_paused_history_removes_conflict_questions_and_ids(self):
        controller, _, _ = self.make_handoff(conflict=True)
        paused = await controller.submit(ORIGINAL)
        self.assertEqual(paused.status, "needs_clarification")
        self.assertTrue(controller.snapshot()["pending_questions"][0]["related_item_ids"])
        result = await controller.submit(controller.snapshot()["confirmed_request"], history=None)
        self.assertEqual(result.status, "ready", result.reason)
        self.assertEqual(result.metadata["workflow"], "normal")
        self.assertEqual(result.metadata["dialogue"]["pending_questions"], [])
        self.assertEqual(result.metadata["dialogue"]["updates"], [])
        self.assertIsNone(controller.optimizer._dialogue_handoff_state)
        await controller.close()

    async def test_workflow_switch_keeps_prior_token_usage(self):
        controller, _, _ = self.make_handoff(token_budget=55)
        engine = controller.optimizer
        history, windows = engine.history, engine.context_windows
        engine.history = engine.context_windows = None
        first = await controller.submit(ORIGINAL)
        self.assertEqual(first.status, "ready", first.reason)
        result = await controller.submit(controller.snapshot()["confirmed_request"],
                                         history=history, context_windows=windows)
        self.assertEqual(result.status, "budget_exceeded")
        self.assertEqual(result.metadata["request_count"], 5)
        self.assertEqual(result.metadata["known_total_tokens"], 55)
        self.assertEqual(result.metadata["limits"]["token_budget"], 55)

    async def test_workflow_switch_continues_after_long_activity(self):
        from unittest.mock import patch
        controller, scripts, history_calls = self.make_handoff()
        engine = controller.optimizer
        history, windows = engine.history, engine.context_windows
        engine.history = engine.context_windows = None
        first = await controller.submit(ORIGINAL)
        self.assertEqual(first.status, "ready", first.reason)
        # Simulate activity beyond both former workflow time budgets.
        engine.started = engine.dialogue_session.paused_at - 1000
        original_call = type(history_calls).__call__

        async def delayed_extract(flow, role, system, payload, purpose):
            if payload.get("phase") == "handoff_extract":
                await asyncio.sleep(0.05)
            return await original_call(flow, role, system, payload, purpose)

        with patch.object(type(history_calls), "__call__", delayed_extract):
            result = await controller.submit(controller.snapshot()["confirmed_request"],
                                             history=history, context_windows=windows)
        self.assertEqual(result.status, "ready", result.reason)
        self.assertGreaterEqual(result.metadata["elapsed_seconds"], 1000)
        self.assertEqual(result.metadata["request_count"], 10)
        self.assertTrue(all(call["status"] == "ok" for call in result.metadata["calls"]))
        self.assertEqual(sum(call["purpose"] == "generate" for call in result.metadata["calls"]), 4)
        await controller.close()

    async def test_partial_conflict_answers_bind_each_new_snapshot(self):
        from optimizer_handoff import HistoryOptions, prepare_history
        batches = []
        async def answer(batch):
            batches.append(batch.to_dict())
            question = batch.questions[0]
            state = controller.optimizer._dialogue_handoff_state
            item_ids = {item["item_id"] for item in state.summary["items"]}
            self.assertTrue(set(question.related_item_ids) <= item_ids)
            self.assertTrue(question.id.startswith(f"handoff-r{state.revision}-"))
            return {"answers": [{"question_id": question.id, "text": "负责人明确采用 Alice。"}]}
        controller, _, calls = self.make_handoff(conflict=True, on_questions=answer)
        controller.optimizer.history = prepare_history(text="用户：Alice 或 Bob。\n用户：Alice 或 Bob。",
                                                      options=HistoryOptions(chunk_bytes=35))
        result = await controller.submit(ORIGINAL)
        self.assertEqual(result.status, "ready", result.reason)
        self.assertEqual(len(batches), 2)
        self.assertNotEqual(batches[0]["questions"][1]["id"], batches[1]["questions"][0]["id"])
        phases = [call[2]["phase"] for call in calls.calls]
        self.assertEqual(phases.count("handoff_update"), 2)
        self.assertEqual(phases.count("handoff_extract"), 2)
        statuses = [item["status"] for item in result.metadata["handoff"]["snapshot"]["summary"]["items"]]
        self.assertNotIn("conflict", statuses)
        self.assertEqual(statuses.count("superseded"), 2)
        await controller.close()

    async def test_changed_windows_are_checked_before_any_new_model_call(self):
        from optimizer_handoff import WindowLimits
        controller, scripts, _ = self.make_handoff()
        first = await controller.submit(ORIGINAL)
        before = len(scripts.calls)
        windows = {"version": 1, "roles": {role: {**value, "context_window": 10000 if role == "a" else value["context_window"]}
                                           for role, value in controller.optimizer.context_windows["roles"].items()}}
        second = await controller.submit(controller.snapshot()["confirmed_request"], context_windows=windows)
        self.assertEqual(second.status, "context_exceeded", second.reason)
        self.assertEqual(len(scripts.calls), before, "new window must reject clarification before the service is called")
        self.assertEqual(second.metadata["request_count"], first.metadata["request_count"])
        self.assertEqual(controller.optimizer.windows.limits,
                         WindowLimits.from_config(windows, controller.optimizer.configs).limits)

    async def test_explicit_session_budget_and_model_configuration_are_fixed(self):
        from dataclasses import replace
        controller, scripts, _ = self.make_handoff()
        first = await controller.submit(ORIGINAL)
        before = len(scripts.calls)
        original_options = controller.optimizer.options
        for changed in (replace(original_options, token_budget=500),
                        replace(original_options, retries=0)):
            controller.optimizer.options = changed
            with self.assertRaisesRegex(ConfigurationError, "不能更换"):
                await controller.submit("补充验收条件")
        controller.optimizer.options = original_options
        original_config = controller.optimizer.configs["a"]
        controller.optimizer.configs["a"] = replace(original_config, name="other-model")
        with self.assertRaisesRegex(ConfigurationError, "不能更换"):
            await controller.submit("补充验收条件")
        controller.optimizer.configs["a"] = original_config
        self.assertEqual(len(scripts.calls), before)
        self.assertTrue(controller.is_current(first))
        self.assertFalse(controller.snapshot()["closed"])
        await controller.close()

    async def test_conflict_answer_updates_snapshot_and_keeps_old_evidence(self):
        async def answer(batch):
            item = batch.questions[0]
            self.assertEqual(item.source, "handoff")
            self.assertTrue(item.related_item_ids)
            return {"answers": [{"question_id": item.id, "text": "负责人明确采用 Alice，其余只读要求不变。"}]}

        controller, scripts, history_calls = self.make_handoff(conflict=True, on_questions=answer,
                                                              clarification=[sufficient("conflict_resolution")])
        result = await controller.submit(ORIGINAL)
        self.assertEqual(result.status, "ready", result.reason)
        snapshot = result.metadata["handoff"]["snapshot"]
        statuses = [item["status"] for item in snapshot["summary"]["items"]]
        self.assertIn("superseded", statuses)
        self.assertIn("confirmed", statuses)
        self.assertEqual([call[2]["phase"] for call in history_calls.calls if call[2]["phase"] != "handoff_fidelity"], ["handoff_extract", "handoff_update"])
        self.assertIn("不改文件", result.optimized_prompt)
        self.assertEqual(snapshot["current_request_sha256"], result.metadata["input_sha256"])
        usage = result.metadata["handoff"]["usage"]
        self.assertTrue(any(call["purpose"].startswith("update-") for call in usage))
        self.assertEqual(result.metadata["request_count"], len(scripts.calls))
        await controller.close()

    async def test_detail_is_incremental_goal_scope_and_uncertain_rebuild_history(self):
        for kind in ("goal_change", "scope_expansion", "uncertain"):
            controller, scripts, history_calls = self.make_handoff(clarification=[sufficient(), sufficient(), sufficient(kind)])
            first = await controller.submit(ORIGINAL)
            self.assertEqual(first.status, "ready", first.reason)
            second = await controller.submit("补充：输出 Markdown 表格。")
            self.assertEqual(second.status, "ready", second.reason)
            self.assertFalse(controller.is_current(first))
            self.assertEqual([call[2]["phase"] for call in history_calls.calls if call[2]["phase"] != "handoff_fidelity"], ["handoff_extract", "handoff_update"])
            previous_hash = second.metadata["handoff"]["snapshot"]["snapshot_sha256"]
            third = await controller.submit("本轮更换任务方向，请重新整理。")
            self.assertEqual(third.status, "ready", third.reason)
            self.assertEqual([call[2]["phase"] for call in history_calls.calls if call[2]["phase"] != "handoff_fidelity"][-1], "handoff_extract")
            self.assertNotEqual(previous_hash, third.metadata["handoff"]["snapshot"]["snapshot_sha256"])
            self.assertFalse(controller.is_current(second))
            await controller.close()

    async def test_history_changes_force_full_even_when_update_classified_detail(self):
        from optimizer_handoff import prepare_history
        controller, _, history_calls = self.make_handoff()
        first = await controller.submit(ORIGINAL)
        replacement = prepare_history(text="用户：新的历史依据，只读，不改文件。")
        second = await controller.submit(controller.snapshot()["confirmed_request"], history=replacement)
        self.assertEqual(second.status, "ready", second.reason)
        self.assertEqual([call[2]["phase"] for call in history_calls.calls if call[2]["phase"] != "handoff_fidelity"], ["handoff_extract", "handoff_extract"])
        self.assertFalse(controller.is_current(first))
        await controller.close()

    async def test_quality_history_usage_contains_incremental_fidelity_and_reuses_c_role(self):
        controller, _, _ = self.make_handoff()
        first = await controller.submit(ORIGINAL)
        self.assertEqual(first.status, "ready", first.reason)
        second = await controller.submit("补充：用 Markdown。")
        self.assertEqual(second.status, "ready", second.reason)
        purposes = [call["purpose"] for call in second.metadata["handoff"]["usage"]]
        self.assertTrue(any(purpose.startswith("update-") and purpose.endswith("_summarize") for purpose in purposes))
        self.assertTrue(any(purpose.startswith("update-") and purpose.endswith("_fidelity") for purpose in purposes))
        await controller.close()


if __name__ == "__main__":
    unittest.main()
