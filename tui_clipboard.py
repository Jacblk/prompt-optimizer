"""Native Unicode clipboard support without a subprocess or extra dependency."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys


class ClipboardError(RuntimeError):
    pass


def copy_windows_text(text: str, *, user32=None, kernel32=None) -> None:
    """Copy UTF-16 text. SetClipboardData owns the allocation only on success."""
    if not isinstance(text, str):
        raise TypeError("clipboard text must be a string")
    if "\x00" in text:
        raise ClipboardError("文本包含剪贴板不支持的空字符。")
    if user32 is None or kernel32 is None:
        if sys.platform != "win32":
            raise ClipboardError("当前平台不能使用 Windows 剪贴板。")
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    signatures = (
        (user32.CreateWindowExW, [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
                                wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, wintypes.HWND, wintypes.HMENU,
                                wintypes.HINSTANCE, ctypes.c_void_p], wintypes.HWND),
        (user32.DestroyWindow, [wintypes.HWND], wintypes.BOOL),
        (user32.OpenClipboard, [wintypes.HWND], wintypes.BOOL),
        (user32.EmptyClipboard, [], wintypes.BOOL),
        (user32.SetClipboardData, [wintypes.UINT, wintypes.HANDLE], wintypes.HANDLE),
        (user32.CloseClipboard, [], wintypes.BOOL),
        (kernel32.GlobalAlloc, [wintypes.UINT, ctypes.c_size_t], wintypes.HGLOBAL),
        (kernel32.GlobalLock, [wintypes.HGLOBAL], ctypes.c_void_p),
        (kernel32.GlobalUnlock, [wintypes.HGLOBAL], wintypes.BOOL),
        (kernel32.GlobalFree, [wintypes.HGLOBAL], wintypes.HGLOBAL),
    )
    for function, argument_types, result_type in signatures:
        function.argtypes = argument_types
        function.restype = result_type

    data = text.encode("utf-16-le") + b"\x00\x00"
    # A private message-only STATIC window works with ConPTY as well as consoles.
    # OpenClipboard(NULL) + EmptyClipboard would leave no clipboard owner.
    owner = user32.CreateWindowExW(0, "STATIC", "", 0, 0, 0, 0, 0, wintypes.HWND(-3), None, None, None)
    if not owner:
        raise ClipboardError("无法创建剪贴板窗口。")
    allocation = kernel32.GlobalAlloc(0x0002, len(data))  # GMEM_MOVEABLE
    if not allocation:
        user32.DestroyWindow(owner)
        raise ClipboardError("无法分配剪贴板内存。")
    opened = False
    transferred = False
    try:
        pointer = kernel32.GlobalLock(allocation)
        if not pointer:
            raise ClipboardError("无法锁定剪贴板内存。")
        try:
            ctypes.memmove(pointer, data, len(data))
        finally:
            kernel32.GlobalUnlock(allocation)
        if not user32.OpenClipboard(owner):
            raise ClipboardError("剪贴板正在被其他程序使用，请稍后重试。")
        opened = True
        if not user32.EmptyClipboard():
            raise ClipboardError("无法更新剪贴板。")
        if not user32.SetClipboardData(13, allocation):  # CF_UNICODETEXT
            raise ClipboardError("无法写入剪贴板。")
        transferred = True
    finally:
        if opened:
            user32.CloseClipboard()
        if not transferred:
            kernel32.GlobalFree(allocation)
        user32.DestroyWindow(owner)
