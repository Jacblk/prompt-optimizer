"""Standalone configuration UI; CLI configuration lives in the existing cli.py."""
from pathlib import Path
import sys

from optimizer_io import configure_stdio


def main(argv=None, *, root=Path(__file__).resolve().parent):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments in (["--help"], ["-h"]):
        print("提示词优化器配置：直接运行打开表单。\n"
              "CLI 配置：prompt-optimizer.cmd --configure / --config-show / --config-check / --config-set KEY VALUE")
        return 0
    if arguments:
        print("此入口打开配置表单；CLI 参数请使用 prompt-optimizer.cmd。", file=sys.stderr)
        return 2
    try:
        from optimizer_config_ui import ConfigurationApp
        result = ConfigurationApp(root=root).run()
        if isinstance(result, str):
            print(result, file=sys.stderr)
            return 1
        return 0
    except (ImportError, OSError, RuntimeError):
        print("配置界面无法启动，请检查项目环境和 requirements.txt。", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    configure_stdio()
    raise SystemExit(main())
