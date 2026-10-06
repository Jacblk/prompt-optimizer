"""Offline terminal/Unicode clipboard probe. Never loads project model settings.

Terminal: paste the EXPECTED string printed before launch, press F2, then Ctrl+Q.
Clipboard mode temporarily replaces text; the caller must preserve/restore their
clipboard. The probe emits only a Boolean match, never the old clipboard text.
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
from ctypes import wintypes
import faulthandler
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPECTED = "窗口粘贴第一行中文\n第二行 {literal}"
CLIPBOARD_SAMPLE = EXPECTED + " 🧪 😀"


def resize_windows_terminal(width, height):
    """Resize this probe's own console buffer/window, never another terminal."""
    class Coord(ctypes.Structure):
        _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]
    class Rect(ctypes.Structure):
        _fields_ = [(name, wintypes.SHORT) for name in ("Left", "Top", "Right", "Bottom")]
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.argtypes, kernel32.GetStdHandle.restype = [wintypes.DWORD], wintypes.HANDLE
    kernel32.SetConsoleWindowInfo.argtypes = [wintypes.HANDLE, wintypes.BOOL, ctypes.POINTER(Rect)]
    kernel32.SetConsoleWindowInfo.restype = wintypes.BOOL
    kernel32.SetConsoleScreenBufferSize.argtypes = [wintypes.HANDLE, Coord]
    kernel32.SetConsoleScreenBufferSize.restype = wintypes.BOOL
    handle = kernel32.GetStdHandle(-11)
    tiny = Rect(0, 0, 1, 1)
    kernel32.SetConsoleWindowInfo(handle, True, ctypes.byref(tiny))
    resized = kernel32.SetConsoleScreenBufferSize(handle, Coord(width, height))
    area = Rect(0, 0, width - 1, height - 1)
    expanded = kernel32.SetConsoleWindowInfo(handle, True, ctypes.byref(area))
    return bool(resized and expanded)


def clipboard_probe():
    from tui_clipboard import copy_windows_text
    copy_windows_text(CLIPBOARD_SAMPLE)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    for function, args, result in (
        (user32.OpenClipboard, [wintypes.HWND], wintypes.BOOL),
        (user32.GetClipboardData, [wintypes.UINT], wintypes.HANDLE),
        (user32.CloseClipboard, [], wintypes.BOOL),
        (kernel32.GlobalLock, [wintypes.HGLOBAL], ctypes.c_void_p),
        (kernel32.GlobalUnlock, [wintypes.HGLOBAL], wintypes.BOOL),
    ):
        function.argtypes, function.restype = args, result
    if not user32.OpenClipboard(None):
        raise RuntimeError("cannot open clipboard for verification")
    try:
        handle = user32.GetClipboardData(13)
        pointer = kernel32.GlobalLock(handle) if handle else None
        if not pointer:
            raise RuntimeError("cannot lock Unicode clipboard text")
        try:
            matches = ctypes.wstring_at(pointer) == CLIPBOARD_SAMPLE
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()
    print(json.dumps({"unicode_clipboard_roundtrip": matches}))
    return 0 if matches else 1


def make_app(root, *, terminal=False, diagnostics=None):
    from optimizer_config import ModelConfig
    from optimizer_engine import Optimizer
    from optimizer_models import ModelReply
    from tui import OptimizerApp

    submitted = []
    class OfflineModel:
        def __init__(self, config):
            self.role = config.role
        def disable_streaming(self):
            self.streaming = False
        async def complete(self, system, payload, *, timeout, on_activity=None):
            if payload.get("phase") == "dialogue_clarification":
                data = {"status": "sufficient", "questions": [], "change_kind": "detail", "reason": "offline"}
            elif self.role == "judge":
                candidates = payload["candidates"]
                chosen = next(item["candidate_id"] for item in candidates
                              if item["candidate_id"] != payload["original_candidate_id"])
                data = {"reviews": [{"candidate_id": item["candidate_id"], "verdict": "pass", "findings": [],
                                     "clarity": 4, "conciseness": 4, "reason": "offline"} for item in candidates],
                        "action": "select", "candidate_id": chosen, "clarification_questions": [], "reason": "offline"}
            else:
                if self.role == "a":
                    submitted.append(payload["original_request"])
                data = {"status": "ready", "optimized_prompt": payload["original_request"],
                        "preserved_constraints": [], "clarification_questions": [], "change_summary": []}
            return ModelReply(json.dumps(data, ensure_ascii=False), 5, 3, 8)
        async def close(self):
            pass
    configs = {role: ModelConfig(role, "offline-" + role, "https://offline.invalid/v1", "placeholder")
               for role in ("a", "b", "judge")}
    async def diagnose(app):
        while True:
            await asyncio.sleep(5)
            tasks = []
            for task in asyncio.all_tasks():
                chain, coro = [], task.get_coro()
                for _ in range(24):
                    frame = getattr(coro, "cr_frame", getattr(coro, "gi_frame", None))
                    if frame is not None:
                        chain.append([frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name])
                    coro = getattr(coro, "cr_await", getattr(coro, "gi_yieldfrom", None))
                    if coro is None:
                        break
                tasks.append({"name": task.get_name(), "chain": chain})
            data = {"closing": app._closing, "busy": app.busy, "report_locked": app._report_lock.locked(),
                    "workers": [[w.name, str(w.state)] for w in app.workers], "tasks": tasks}
            diagnostics.with_suffix(".tasks.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    class ProbeApp(OptimizerApp):
        CSS_PATH = str(Path(__file__).resolve().parents[1] / "tui.tcss")
        probe_sizes = []
        probe_resizes = []
        def on_resize(self, event):
            super().on_resize(event)
            self.probe_sizes.append((event.size.width, event.size.height))
        def on_mount(self):
            if terminal:
                self.set_timer(90.0, self.action_quit)
                if sys.platform == "win32":
                    def resize(width, height):
                        self.probe_resizes.append((width, height, resize_windows_terminal(width, height)))
                    self.set_timer(0.2, lambda: resize(100, 30))
                    self.set_timer(0.8, lambda: resize(80, 24))
            if diagnostics:
                asyncio.create_task(diagnose(self))
        def _observe_result(self, result):
            super()._observe_result(result)
            if terminal and result.status in {"ready", "unreviewed"}:
                self.set_timer(1.0, self.action_quit)
    app = ProbeApp(root=root, config_loader=lambda root: configs,
                       optimizer_factory=lambda configs, options: Optimizer(configs, options, factory=OfflineModel))
    return app, submitted


async def screenshots(directory):
    from textual.widgets import TextArea
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        app, _ = make_app(Path(folder))
        async with app.run_test(size=(120, 42)) as pilot:
            app.query_one("#message-input", TextArea).load_text(EXPECTED)
            await pilot.pause()
            (directory / "tui-120x42.svg").write_text(app.export_screenshot(), encoding="utf-8")
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            (directory / "tui-80x24.svg").write_text(app.export_screenshot(), encoding="utf-8")
    print(json.dumps({"screenshots": [str(p) for p in directory.glob("tui-*.svg")]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--clipboard", action="store_true")
    group.add_argument("--screenshots", type=Path)
    group.add_argument("--terminal", action="store_true")
    parser.add_argument("--diagnostics", type=Path, help="Optional offline-only thread traceback file")
    args = parser.parse_args()
    if args.clipboard:
        return clipboard_probe()
    if args.screenshots:
        asyncio.run(screenshots(args.screenshots))
        return 0
    print("EXPECTED synthetic paste: " + EXPECTED)
    print("Paste, then press F2 separately. The probe exits after the result.")
    with tempfile.TemporaryDirectory() as folder:
        app, submitted = make_app(Path(folder), terminal=True, diagnostics=args.diagnostics)
        diagnostic = args.diagnostics.open("w", encoding="utf-8") if args.diagnostics else None
        if diagnostic:
            faulthandler.dump_traceback_later(8, repeat=True, file=diagnostic)
        try:
            app.run()
        finally:
            if diagnostic:
                faulthandler.cancel_dump_traceback_later()
                diagnostic.close()
        outcome = {"terminal_paste_match": submitted == [EXPECTED],
                   "status": app.result.status if app.result else None,
                   "reports_saved": len(app.report_paths),
                   "terminal_resize_requests": app.probe_resizes,
                   "terminal_resize_events": app.probe_sizes,
                   "terminal_resize_verified": (100, 30) in app.probe_sizes and (80, 24) in app.probe_sizes}
        print(json.dumps(outcome))
        return 0 if outcome["terminal_paste_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
