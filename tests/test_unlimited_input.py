"""Long input contracts using local fixtures and scripted models only."""
from __future__ import annotations

import hashlib
from pathlib import Path
import random
import tempfile
import unittest

from optimizer_config import ConfigurationError
from optimizer_documents import ReferenceFile, ReferenceOptions, prepare_references
from optimizer_engine import Optimizer, RunOptions
from optimizer_handoff import WindowExceeded
from test_dialogue import DialogueScripts, ask, make_controller, question, sufficient
from test_dialogue_handoff import ScriptedCalls, pipeline, updated
from test_optimizer import ORIGINAL, Scripts, configs, draft, layer_analysis, review


class LongReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def file(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8", newline="")
        return path

    def test_full_default_keeps_large_file_and_final_marker(self):
        text = "访谈开头。\n" + "人物、时间与事件须完整保留。\n" * 8000 + "访谈最后一段。\n"
        bundle = prepare_references([ReferenceFile(self.file("访谈.txt", text))])
        self.assertGreater(bundle.chars, 100000)
        self.assertEqual(bundle.files[0].extracted_chars, len(text))
        self.assertEqual(len(bundle.files[0].selected), len(bundle.files[0].chunks))
        self.assertEqual(bundle.files[0].selected[-1].metadata["end_index"], len(text))
        self.assertIn("访谈最后一段。", bundle.blocks[0])
        self.assertIsNone(bundle.metadata()["options"]["max_chars"])

    def test_relevant_default_can_select_over_20000_characters(self):
        files = [ReferenceFile(self.file(f"reference-{index}.txt", "alpha beta\n" * 4500))
                 for index in range(2)]
        bundle = prepare_references(files, ReferenceOptions(mode="relevant", chunk_size=6000,
                                                            chunk_overlap=0), query="alpha")
        self.assertGreater(bundle.chars, 20000)
        self.assertEqual(sum(len(file.selected) for file in bundle.files), 6)
        self.assertTrue(all(file.selected for file in bundle.files))
        self.assertTrue(bundle.warnings)
        self.assertIn("未覆盖全文", bundle.blocks[0])

    def test_explicit_character_caps_still_reject_without_truncation(self):
        path = self.file("reference.txt", "alpha beta\n" * 3000)
        for mode in ("full", "relevant"):
            with self.subTest(mode=mode), self.assertRaises(ConfigurationError):
                prepare_references([ReferenceFile(path)], ReferenceOptions(mode=mode, max_chars=10),
                                   query="alpha")

    def test_optional_caps_reject_invalid_values(self):
        for invalid in (0, -1, True, 1.5, "100"):
            for option, field in ((ReferenceOptions, "max_chars"), (RunOptions, "max_input_chars")):
                with self.subTest(option=option.__name__, value=invalid), self.assertRaises(ConfigurationError):
                    option(**{field: invalid})


class LongWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_large_material_survives_generation_original_review_and_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.txt"
            path.write_text("alpha 访谈内容。\n" * 10000 + "最后一段。\n", encoding="utf-8")
            bundle = prepare_references([ReferenceFile(path)])
            scripts = Scripts(judge=[lambda payload: review(payload, "repair"), review],
                              a=[draft("甲提示词草稿。"), draft(bundle.attach(ORIGINAL))], b=[draft(ORIGINAL)])
            engine = Optimizer(configs(), RunOptions(), references=bundle, factory=scripts.factory,
                               rng=random.Random(1))
            result = await engine.run(ORIGINAL)
            self.assertEqual(result.status, "ready", result.reason)
            self.assertEqual(result.optimized_prompt, bundle.attach(ORIGINAL))
            self.assertGreater(len(result.optimized_prompt), 100000)
            self.assertEqual(len(scripts.calls), 5)
            for _, _, payload, _ in scripts.calls:
                self.assertEqual(payload["reference_files"][0]["block"], bundle.blocks[0])
            self.assertTrue(all(candidate["optimized_prompt"].count(bundle.blocks[0]) == 1
                                for candidate in result.candidates))

    async def test_long_initial_dialogue_request_is_preserved_in_all_candidates(self):
        text = ORIGINAL + "\n背景输入：\n" + "保留这段原始输入。\n" * 11000
        controller, scripts = make_controller()
        try:
            result = await controller.submit(text)
            self.assertEqual(result.status, "ready", result.reason)
            self.assertGreater(len(text), 100000)
            self.assertEqual(result.optimized_prompt, text)
            self.assertTrue(all(candidate["optimized_prompt"] == text for candidate in result.candidates))
            self.assertTrue(all(call[2]["original_request"] == text for call in scripts.calls))
        finally:
            await controller.close()

    async def test_long_followup_preserves_existing_request_and_entire_update(self):
        controller, scripts = make_controller()
        followup = "补充访谈内容：\n" + "人物、时间与事件。\n" * 6000
        try:
            first = await controller.submit(ORIGINAL)
            result = await controller.submit(followup)
            self.assertEqual(result.status, "ready", result.reason)
            self.assertGreater(len(followup), 50000)
            confirmed = controller.snapshot()["confirmed_request"]
            self.assertIn(ORIGINAL, confirmed)
            self.assertIn(followup, confirmed)
            self.assertEqual(result.optimized_prompt, confirmed)
            self.assertEqual(scripts.calls[-1][2]["original_request"], confirmed)
            self.assertFalse(controller.is_current(first))
        finally:
            await controller.close()

    async def test_long_clarification_answer_is_accepted_without_losing_raw_text(self):
        answer = "具体检查对象与验收要求。\n" * 5000

        async def respond(batch):
            return {"answers": [{"question_id": batch.questions[0].id, "text": answer}]}

        controller, _ = make_controller(DialogueScripts(clarification=[ask(question()), sufficient()]),
                                        on_questions=respond)
        try:
            result = await controller.submit(ORIGINAL)
            self.assertEqual(result.status, "ready", result.reason)
            self.assertGreater(len(answer), 50000)
            self.assertIn(answer, controller.snapshot()["confirmed_request"])
            response = result.metadata["dialogue"]["question_rounds"][0]["response"]
            self.assertEqual(response["answers"][0]["text"], answer)
        finally:
            await controller.close()

    async def test_legacy_layer_supplement_is_not_blocked_at_50000_characters(self):
        supplement = "当前输入资料。\n" * 7000
        scripts = Scripts(analysis=[layer_analysis(missing={"context": "资料待补充"})])
        choices = [{"layer": "context", "choice": "user", "value": supplement}]
        engine = Optimizer(configs(), RunOptions(choose_layers=True), factory=scripts.factory,
                           layer_resolver=lambda plan: choices)
        result = await engine.run(ORIGINAL)
        self.assertEqual(result.status, "ready", result.reason)
        self.assertGreater(len(supplement), 50000)
        for _, _, payload, _ in scripts.calls[1:]:
            self.assertEqual(payload["layer_decisions"], choices)

    async def test_explicit_input_cap_still_rejects_before_model_calls(self):
        controller, scripts = make_controller(max_input_chars=10)
        try:
            with self.assertRaises(ConfigurationError):
                await controller.submit("长" * 11)
            self.assertEqual(scripts.calls, [])
        finally:
            await controller.close()


class LongHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_handoff_full_accepts_long_current_request(self):
        request = "继续只读。\n" + "当前输入资料。\n" * 7000
        flow = pipeline(window=1000000)
        flow.options.max_input_chars = RunOptions().max_input_chars
        state = await flow.full(request, revision=1)
        self.assertGreater(len(request), 50000)
        self.assertEqual(state.snapshot.payload()["current_request_sha256"],
                         hashlib.sha256(request.encode()).hexdigest())
        self.assertIn(request, state.snapshot.attach("提示词草稿。", request))

    async def test_handoff_update_keeps_long_answer_as_versioned_evidence(self):
        answer = "只读检查输入资料。\n" * 6000

        def summarized_update(payload):
            data = updated(payload)
            doc = payload["stage_input"]["chunks"][0]
            data["items"][-1]["text"] = "只读检查输入资料。"
            data["items"][-1]["citations"][0]["quote"] = doc["text"].splitlines()[0]
            return data

        calls = ScriptedCalls(update=summarized_update)
        flow = pipeline(calls, window=1000000)
        flow.options.max_input_chars = RunOptions().max_input_chars
        base = await flow.full("继续只读。", revision=1)
        updated_request = "继续只读。\n" + answer
        state = await flow.update(base, updated_request,
                                  [{"question_id": "scope", "text": answer, "related_item_ids": []}],
                                  revision=2)
        self.assertGreater(len(answer), 50000)
        self.assertEqual(state.evidence["answer-r2-scope"]["text"], answer)
        self.assertEqual(state.evidence["request-r2"]["text"], updated_request)
        self.assertEqual(calls.calls[-2][2]["phase"], "handoff_update")
        self.assertEqual(base.evidence["request-r1"]["text"], "继续只读。")

    async def test_model_window_still_rejects_long_request_before_any_call(self):
        calls = ScriptedCalls()
        flow = pipeline(calls, window=32000)
        flow.options.max_input_chars = RunOptions().max_input_chars
        with self.assertRaises(WindowExceeded):
            await flow.full("当前输入资料。\n" * 7000, revision=1)
        self.assertEqual(calls.calls, [])
