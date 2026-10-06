import ctypes
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tui_clipboard import ClipboardError, copy_windows_text


class ClipboardTests(unittest.TestCase):
    def api(self):
        self.buffer = ctypes.create_string_buffer(1024)
        user32 = SimpleNamespace(
            CreateWindowExW=Mock(return_value=0x1_0000_0001),
            DestroyWindow=Mock(return_value=True),
            OpenClipboard=Mock(return_value=True), EmptyClipboard=Mock(return_value=True),
            SetClipboardData=Mock(return_value=0x2_0000_0002), CloseClipboard=Mock(return_value=True),
        )
        kernel32 = SimpleNamespace(
            GlobalAlloc=Mock(return_value=0x2_0000_0002),
            GlobalLock=Mock(return_value=ctypes.addressof(self.buffer)),
            GlobalUnlock=Mock(return_value=True), GlobalFree=Mock(return_value=0),
        )
        return user32, kernel32

    def test_unicode_multiline_and_64_bit_handles_transfer_to_system(self):
        user32, kernel32 = self.api()
        text = "中文第一行\n第二行 {payload} 😀"
        copy_windows_text(text, user32=user32, kernel32=kernel32)
        expected = text.encode("utf-16-le") + b"\x00\x00"
        self.assertEqual(self.buffer.raw[:len(expected)], expected)
        user32.OpenClipboard.assert_called_once_with(0x1_0000_0001)
        user32.SetClipboardData.assert_called_once_with(13, 0x2_0000_0002)
        user32.CloseClipboard.assert_called_once()
        user32.DestroyWindow.assert_called_once_with(0x1_0000_0001)
        kernel32.GlobalFree.assert_not_called()
        self.assertIs(user32.CreateWindowExW.restype, ctypes.c_void_p)
        self.assertIs(kernel32.GlobalLock.restype, ctypes.c_void_p)

    def test_failed_transfer_closes_window_and_reclaims_memory(self):
        user32, kernel32 = self.api()
        user32.SetClipboardData.return_value = 0
        with self.assertRaises(ClipboardError):
            copy_windows_text("失败后不误报成功", user32=user32, kernel32=kernel32)
        user32.CloseClipboard.assert_called_once()
        user32.DestroyWindow.assert_called_once()
        kernel32.GlobalFree.assert_called_once_with(0x2_0000_0002)

    def test_busy_clipboard_reclaims_memory_without_close(self):
        user32, kernel32 = self.api()
        user32.OpenClipboard.return_value = False
        with self.assertRaises(ClipboardError):
            copy_windows_text("中文", user32=user32, kernel32=kernel32)
        user32.CloseClipboard.assert_not_called()
        user32.SetClipboardData.assert_not_called()
        kernel32.GlobalFree.assert_called_once()
        user32.DestroyWindow.assert_called_once()

    def test_lock_and_allocation_failure_clean_up(self):
        for failed in ("GlobalAlloc", "GlobalLock"):
            with self.subTest(failed=failed):
                user32, kernel32 = self.api()
                getattr(kernel32, failed).return_value = 0
                with self.assertRaises(ClipboardError):
                    copy_windows_text("中文", user32=user32, kernel32=kernel32)
                user32.DestroyWindow.assert_called_once()
                user32.OpenClipboard.assert_not_called()
                self.assertEqual(kernel32.GlobalFree.call_count, int(failed == "GlobalLock"))

    def test_null_character_is_rejected_before_clipboard_changes(self):
        user32, kernel32 = self.api()
        with self.assertRaises(ClipboardError):
            copy_windows_text("a\x00b", user32=user32, kernel32=kernel32)
        user32.CreateWindowExW.assert_not_called()
        kernel32.GlobalAlloc.assert_not_called()

    def test_owner_creation_failure_does_not_touch_clipboard_or_allocate(self):
        user32, kernel32 = self.api()
        user32.CreateWindowExW.return_value = 0
        with self.assertRaises(ClipboardError):
            copy_windows_text("原剪贴板应保留", user32=user32, kernel32=kernel32)
        kernel32.GlobalAlloc.assert_not_called()
        user32.OpenClipboard.assert_not_called()
        user32.EmptyClipboard.assert_not_called()
        user32.DestroyWindow.assert_not_called()

    def test_failed_empty_closes_window_and_frees_without_transferring(self):
        user32, kernel32 = self.api()
        user32.EmptyClipboard.return_value = False
        with self.assertRaises(ClipboardError):
            copy_windows_text("更新失败不误报成功", user32=user32, kernel32=kernel32)
        user32.SetClipboardData.assert_not_called()
        user32.CloseClipboard.assert_called_once()
        user32.DestroyWindow.assert_called_once_with(0x1_0000_0001)
        kernel32.GlobalFree.assert_called_once_with(0x2_0000_0002)

    def test_non_text_is_rejected_before_any_system_changes(self):
        user32, kernel32 = self.api()
        with self.assertRaises(TypeError):
            copy_windows_text(None, user32=user32, kernel32=kernel32)
        user32.CreateWindowExW.assert_not_called()
        kernel32.GlobalAlloc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
