# 旧流程回归夹具

`legacy_optimize.py` 和 `legacy_launcher.py` 保留旧版 CLI 参数和菜单输入接口，仅供既有离线测试调用。两者没有可执行入口，也不被产品模块导入。

这些夹具的旧模式参数和菜单按键映射到当前固定 A/B 生成与独立评审；不提供产品快速模式或换序复核。删除运行模式后的调用数断言已迁移，文件材料、原文候选和修复检查继续有效。修改前的完整源码保存在 `baselines/before_four_layers_auto_workflow_20261004_195351/`（仅本地保留），历史报告不重写。

这些测试继续检查底层生成、评审、文件材料、预算和旧报告兼容。当前用户入口包括 `prompt-optimizer.cmd` / `cli.py` 的 CLI，以及 `启动提示词优化器.cmd` / `launcher.py` 的 TUI。入口测试分别位于 `test_cli.py` 和 `test_tui_launcher.py`。
