# 维护说明

先阅读 [README](README.md) 的当前 CLI/TUI 行为和[兼容性记录](COMPATIBILITY.md)。在项目 `.venv` 中安装 `requirements.txt` 固定版本；调整版本前核对实际导入、传递依赖和兼容性，记录验证环境。

```powershell
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -B -X utf8 -W ignore::DeprecationWarning -m unittest discover -s tests -q
& .\.venv\Scripts\python.exe -B -X utf8 tools\prepare_repository.py
```

回归测试使用模拟模型、临时配置和本机回环接口。`tests/legacy_optimize.py`、`tests/legacy_launcher.py` 及九份公开基线支撑历史兼容与开发对照，清理时先核对引用。[公开文件规则](tools/repository_policy.json)控制上传范围；新增文件时更新规则及 `.gitignore` 并运行预检。

真实 `.env` 不用于审计、搜索或打包。不要在源码、测试夹具、截图或文档里写入真实密钥、个人路径、任务材料或运行报告；本地测试输出放在被忽略的 `maintenance/`、`reports/` 或 `evals/results/`。真实模型验证需要明确授权，保留实际用量及未知结果，不能用离线通过替代模型效果结论。

提交说明写清问题、最终行为和验证证据。更改 CLI 参数、TUI 行为、保存契约或模型调用流程时同步更新相应使用说明。当前没有选定开源许可证；发布开源时由所有者确定。
