"""Start the conversational TUI; cli.py provides the terminal interface."""
from __future__ import annotations

from pathlib import Path
import sys

from optimizer_io import configure_stdio

ROOT = Path(__file__).resolve().parent


def main(argv=None, *, stdout=None, stderr=None, run_tui=None, root=ROOT):
    arguments = list(sys.argv[1:] if argv is None else argv)
    output = stdout if stdout is not None else sys.stdout
    errors = stderr if stderr is not None else sys.stderr
    if arguments in (["--help"], ["-h"]):
        print("对话式提示词优化器\n直接运行即可进入 TUI，或双击 启动提示词优化器.cmd。\n"
              "固定 A/B 生成、C 独立评审；历史有内容时自动交接，清空后普通生成。\n"
              "预算、材料、历史和报告均在界面中管理。\n"
              "命令行接口：prompt-optimizer.cmd --help（或 python cli.py --help）。", file=output)
        return 0
    if arguments:
        print("此入口只使用 TUI，不再接受旧命令行参数。请直接启动，在界面中输入需求；历史有内容时自动交接。\n"
              "使用 CLI 请运行 prompt-optimizer.cmd --help（或 python cli.py --help）。",
              file=errors)
        return 2
    try:
        if run_tui is None:
            from tui import main as run_tui
        result = run_tui(root=Path(root))
        return 0 if result is None else int(result)
    except ModuleNotFoundError:
        project_root = Path(root).resolve()
        python_path = project_root / ".venv" / "Scripts" / "python.exe"
        print("界面依赖尚未安装完整。只需为当前环境安装一次：\n"
              f'& "{python_path}" -m pip install -r "{project_root / "requirements.txt"}"',
              file=errors)
        return 1
    except (ImportError, OSError, RuntimeError):
        print("对话式界面未能启动；请检查项目依赖和终端后重试。", file=errors)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    configure_stdio()
    raise SystemExit(main())
