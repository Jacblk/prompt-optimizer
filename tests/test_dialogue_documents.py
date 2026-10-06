"""Cached material reselection stays local and leaves previous artifacts stable."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from optimizer_config import ConfigurationError
from optimizer_documents import (
    LocalReferenceLoader, PARTIAL_REFERENCE_WARNING, ReferenceFile, ReferenceOptions,
    prepare_references, reselect_references,
)


class CachedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "材料.md"
        self.path.write_text("alpha 订单缓存方案。\n\n" + "填充。" * 45 + "\n\n"
                             + "beta 退款处理方案。\n\n" + "占位。" * 45, encoding="utf-8")

    def bundle(self, mode="relevant"):
        return prepare_references([ReferenceFile(self.path)],
                                  ReferenceOptions(mode=mode, chunk_size=80, chunk_overlap=0, max_chunks=1), query="alpha")

    def test_new_query_uses_cached_complete_chunks_without_loading_or_path_checks(self):
        bundle = self.bundle()
        previous = bundle.metadata()
        self.path.unlink()
        with patch.object(LocalReferenceLoader, "load", side_effect=AssertionError("no reload")), \
                patch("optimizer_documents.check_reference_path", side_effect=AssertionError("no disk lookup")):
            new = reselect_references(bundle, query="beta")
        self.assertEqual(bundle.metadata(), previous)
        self.assertNotEqual(new.blocks, bundle.blocks)
        self.assertIn("beta", new.blocks[0])
        self.assertNotIn("alpha", new.blocks[0])
        self.assertEqual(new.files[0].sha256, bundle.files[0].sha256)
        self.assertEqual(new.files[0].chunks, bundle.files[0].chunks)
        self.assertIsNot(new.files[0].selected, bundle.files[0].selected)

    def test_same_query_keeps_preview_and_scope_identical_and_warning_not_duplicated(self):
        bundle = self.bundle()
        first = reselect_references(bundle, query="alpha")
        second = reselect_references(first, query="alpha")
        self.assertEqual(second.metadata(), bundle.metadata())
        self.assertEqual(second.warnings.count(PARTIAL_REFERENCE_WARNING), 1)

    def test_full_mode_selects_all_cached_chunks_without_keywords(self):
        bundle = self.bundle("full")
        previous = bundle.metadata()
        new = reselect_references(bundle, query="")
        self.assertEqual(new.metadata(), previous)
        self.assertTrue(new.files[0].summary("full")["covers_all_text"])
        self.assertIsNot(new.files[0].selected, bundle.files[0].selected)

    def test_empty_or_unmatched_query_and_small_budget_do_not_change_previous_selection(self):
        bundle = self.bundle()
        previous = bundle.metadata()
        for query in ("", "nonexistent"):
            with self.subTest(query=query), self.assertRaises(ConfigurationError):
                reselect_references(bundle, query=query)
            self.assertEqual(bundle.metadata(), previous)
        limited = replace(bundle, options=replace(bundle.options, max_chars=1))
        with self.assertRaises(ConfigurationError):
            reselect_references(limited, query="beta")
        self.assertEqual(bundle.metadata(), previous)

    def test_pdf_warnings_and_purpose_scope_survive_reselection(self):
        bundle = self.bundle()
        bundle.files[0].empty_pages = [2]
        bundle.files[0].page_count = 3
        bundle.warnings.insert(0, "第 2 页未提取到文字，未做 OCR。")
        new = reselect_references(bundle, query="beta")
        self.assertEqual(new.files[0].empty_pages, [2])
        self.assertIn("未做 OCR", new.blocks[0])
        self.assertEqual(new.files[0].purpose, bundle.files[0].purpose)
        self.assertEqual(new.files[0].usage, bundle.files[0].usage)
        self.assertIn("第 2 页未提取到文字，未做 OCR。", new.warnings)


if __name__ == "__main__":
    unittest.main()
