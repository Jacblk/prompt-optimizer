# 仓库上传准备

公开范围由 [repository_policy.json](tools/repository_policy.json) 定义，`.gitignore` 使用同一规则。预检默认只读；只有显式指定 `--manifest` 或 `--archive` 才在本地 `maintenance/` 下写产物。工具不初始化 Git、不暂存或提交、不创建远端、不推送、不调用模型。

## 上传范围

| 内容 | 上传规则 |
| --- | --- |
| 运行源码、CLI/TUI 启动器、样式、固定依赖、公开窗口配置 | 根目录逐文件白名单 |
| `.env.example` | 公开空凭据模板 |
| README、使用说明、当前兼容性与验收记录、维护说明 | 根目录逐文件白名单 |
| `tests/` | 当前 Python 测试及说明，包含旧回归夹具 |
| `evals/` | 顶层 Python 探针与静态 JSONL 样例 |
| `examples/`、`course_analysis/` | Markdown 模板与课程结论 |
| `.github/workflows/`、`tools/` | 离线 CI、预检工具及规则 |
| `baselines/` | 下列九个精确文件；其他历史快照和证据仅本地保留 |

| 基线文件（均位于 `baselines/`） | 保留原因 |
| --- | --- |
| `system_prompt_v1.txt` | `evaluate.py` 与评测回归 |
| `optimizer_layers_v2_4_0.py.txt` | 真实旧数据结构兼容回归 |
| `optimizer_examples_v2_2_1.py.txt`、`optimizer_prompts_v2_2_1.py.txt` | `evals/live_probe.py` 旧版对照 |
| `optimizer_examples_v2_3.py.txt`、`optimizer_prompts_v2_3.py.txt` | `evals/cot_probe.py` 旧版对照 |
| `layer_boundary_20261005/templates.json` | 边界探针与离线回归的完整提示材料 |
| `optimizer_prompts_v2_0.py.txt`、`optimizer_layers_v2_3.py.txt` | 公开课程结论的历史依据链接 |

真实 `.env` 和它的私有变体、`.venv`、缓存、`reports/`、`last_optimized_prompt.md`、`evals/results/`、`evals/artifacts/`、维护审计和其他历史基线不进入上传清单，也不进入预检内容扫描。真实 `.env` 只查询目录项存在性。排除上传不会删除本地原件。公开文档中的历史运行证据标明“本地保留”，因此源码包不含这些原始记录。

## 本地验证和打包

```powershell
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -B -X utf8 -W ignore::DeprecationWarning -m unittest discover -s tests -q
& .\.venv\Scripts\python.exe -B -X utf8 tools\prepare_repository.py
& .\.venv\Scripts\python.exe -B -X utf8 tools\prepare_repository.py --manifest maintenance/20261006/repository-manifest.json --archive maintenance/20261006/prompt-optimizer-source.zip
```

清单记录公开文件的相对路径、大小和 SHA256，不记录本机根目录或凭据。工具检查缺失文件、链接文件、个人绝对路径、常见密钥形式、失效本地 Markdown 链接、`.gitignore` 漂移及已暂存的白名单外文件。风险输出仅有文件、行号和风险类型；有风险时拒绝生成 ZIP。退出码 `0` 为通过，`1` 为检查发现，`2` 为无法完成操作。扫描只能辅助复核，应查看清单并确认素材与代码可以分享。

ZIP 使用扫描时的同一份公开文件内容，包含全部运行/测试/探针基线，可重新生成。验证源码包时，在解包后的目录重新安装依赖并运行以上检查；Windows CMD 启动器测试要求该目录下存在 `.venv/Scripts/python.exe`。

## 上传步骤

1. 复核 `repository-manifest.json`、源码包及离线验证结果。
2. 确定仓库托管平台、名称、可见性和远端地址。公开开源时由所有者选择许可证，再将许可证文件加入规则；此项目尚未自行选择许可证。
3. 初始化本地 Git 并逐项核对暂存清单；真实配置和个人产物必须继续被忽略。上传前重跑预检，让已暂存的白名单外文件阻止上传。
4. 创建提交并上传到确定的远端，查看远端 Actions 结果。

需要新增根目录文件或公开基线时，先修改 JSON 白名单，再按同一规则更新 `.gitignore`。生成规则可使用以下命令，随后查看文件变更：

```powershell
& .\.venv\Scripts\python.exe -B -X utf8 -c 'from pathlib import Path; from tools.prepare_repository import load_policy, render_gitignore; root = Path.cwd(); (root / ".gitignore").write_text(render_gitignore(load_policy(root)), encoding="utf-8")'
```

## CI 验证边界

[离线 CI](.github/workflows/offline-tests.yml) 仅配置 Windows / Python 3.14：先创建项目 `.venv` 并安装 pins，再做 `pip check`、完整离线回归与前后上传预检。测试使用模拟模型、本机回环接口和临时材料，不读取真实配置或访问外部模型；依赖安装本身需要访问软件包源。CI 配置尚不代表远端已执行成功，也不证明真实模型语义质量。

Action 用法已核对 [checkout 官方说明](https://github.com/actions/checkout)和 [setup-python 官方说明](https://github.com/actions/setup-python)。工作流使用只读仓库权限，关闭 checkout 凭据保留，无需模型密钥。
