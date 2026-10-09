# CLI 使用说明

CLI 与 TUI 共用需求澄清、A/B 生成、C 独立评审、可配置额度的自动修复及报告保存。每次执行处理一项需求，允许在执行期间连续回答澄清；结果生成后退出。无需新增依赖，不修改全局 PATH。

## 启动与输入

Windows 项目目录中运行：

```powershell
# 直接给需求
.\prompt-optimizer.cmd "只读检查 src 中的异常处理，输出问题清单"

# 多行终端输入：单独一行 END 提交
.\prompt-optimizer.cmd

# 从 UTF-8 文件读取，成功后保存到指定文件
.\prompt-optimizer.cmd --input .\request.md --output .\result.md

# 完整帮助；也可直接使用项目 Python
.\prompt-optimizer.cmd --help
& .\.venv\Scripts\python.exe -X utf8 cli.py --help
```

需求来源互斥：位置参数 `TEXT`、`-p/--prompt/--request`、`-i/--input FILE`。`--input -` 或无参数的管道输入读取全部标准输入直到 EOF；管道中的 `END` 是普通文本。需求支持中文、英文、多行及 UTF-8 BOM。配置文件不能作为需求、参考、历史或报告读取。

启动器使用项目 `.venv`，可用绝对路径从其他目录运行。默认 `.env`、`optimizer_settings.json`、`context_windows.json`、`reports/` 和 `last_optimized_prompt.md` 均位于项目目录；显式输入/输出相对路径以调用时的当前目录为基准。三个配置文件仅本地保存，仓库提供空的 `context_windows.example.json`；首次填写并保存窗口时创建本地窗口文件。

## 澄清与脚本调用

stdin 与 stdout 都连接终端时自动允许澄清。每轮最多三个问题，可填写自由文本，或用 `/1`、`/2` 等明确采用展示的建议。`/multi` 输入多行回答，以单独一行 `END` 提交；空行或 `/skip` 跳过该问题，不默认采用建议。

本轮回答统一提交；`/pause` 保存此前已确认内容与未决问题后退出，`/cancel` 取消，本轮尚未提交的回答不生效。Ctrl+C 可取消运行或终端输入；取消和未决状态不覆盖上次成功提示词。无有效回答时保存未决报告并退出，不反复空请求。

管道、stdout 重定向或 `--non-interactive` 禁止等待澄清回答。遇到问题返回退出码 `3`，问题可从 JSON 或报告读取。可以把回答补充进需求后重新执行；CLI 暂不从报告恢复上次进程的会话。使用文件或位置参数提供需求时，`--interactive` 可显式允许在 stdout 重定向的情况下从终端回答。stdin 已用于管道时不能同时用于交互回答。

```powershell
# 适合自动化：生成成功时退出码为 0
.\prompt-optimizer.cmd --input .\request.md --non-interactive --json > .\result.json
$LASTEXITCODE

# PowerShell 管道显式使用 UTF-8，兼容 Windows PowerShell 5.1
$previousEncoding = $OutputEncoding
try {
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    Get-Content -Raw -Encoding UTF8 .\request.md | .\prompt-optimizer.cmd --json
} finally {
    $OutputEncoding = $previousEncoding
}
```

直接使用 `--input` 可避免 shell 管道编码差异。Windows PowerShell 5.1 的 `>` 使用其自身输出编码；需要 UTF-8 提示词/报告文件时使用 `--output`、`--report`，程序会直接以 UTF-8 保存。

## 结果、报告与退出码

stdout 默认仅输出成功提示词；未决或待复核的草稿保留在报告中。`--json` 改为输出完整结果，包括 `status`、`optimized_prompt`、`questions`、`metadata`、`files.report`、`files.prompt`、`save_errors`。错误在参数解析成功后也以 JSON 输出，包含 `status: error`、`reason` 和 `exit_code`。帮助和参数解析错误仍使用普通命令行格式。

进度、问题、保存路径和错误写入 stderr；`--quiet` 隐藏进度与保存提示。运行状态来自共享引擎，不打印模型推理原文或未校验的正文。`--report FILE` 指定报告目标时，等待确认和最终状态依次更新该文件；默认每个检查点单独保存时间戳报告。保存目标不能覆盖输入、配置、材料、历史或项目源文件。

| 退出码 | 含义 |
| --- | --- |
| `0` | 优化成功，或离线查看、预览、配置动作成功 |
| `1` | 运行失败、预算耗尽、交接检查失败或保存失败；保存失败时 stdout 仍可保留成功结果 |
| `2` | 参数、输入或配置无效，尚未开始模型流程 |
| `3` | 需要澄清；报告中保留已确认需求和问题 |
| `4` | 需要复核；未发布为成功提示词 |
| `130` | 用户取消 |

离线回看不会恢复会话，也不会更新成功文件：

```powershell
.\prompt-optimizer.cmd --show-report .\reports\example.json --json
```

## 材料与交接

`--reference-file` 可重复指定本地文本、代码或可提取文字的 PDF。`--reference-purpose`、`--reference-usage` 和 `--reference-kind` 对本次全部参考文件生效。材料模式沿用保存的默认值，初始为全文；`--reference-mode relevant` 可为本次覆盖，使用关键词选块，需提供需求用于匹配，报告会注明部分覆盖。自然语言中的路径、网址、视频或图片不会自动读取。

```powershell
.\prompt-optimizer.cmd --input .\request.md --reference-file .\sample.py --reference-purpose "借鉴代码组织方式" --reference-usage "仅参考结构"
.\prompt-optimizer.cmd --reference-file .\sample.py --preview-references
.\prompt-optimizer.cmd "参考缓存实现" --reference-file .\sample.py --reference-mode relevant --preview-references --json
```

`--history-file` 接受 TXT/Markdown，可重复指定并保留顺序；与 `--history-text` 互斥。有历史时自动交接，历史加载失败不会跳过历史继续生成。交接使用与当前模型身份匹配的窗口配置；缺少或不匹配时先填写服务商提供的真实窗口大小。以下数字仅演示参数顺序：

```powershell
# A、B、C 的上下文窗口；只校验并保存公开绑定，不请求模型
.\prompt-optimizer.cmd --configure-contexts 128000 128000 128000
.\prompt-optimizer.cmd "继续只读检查" --history-file .\history.md
```

`--configure-contexts` 保留兼容语法，内部使用统一窗口绑定和保存服务。有效窗口直接复用，缺失、绑定失效或容量不足时零模型调用并返回配置错误；配置完成后重新运行生成命令。可用 `--context-config FILE` 指定其他窗口配置。材料与交接的覆盖、来源、SHA256、权限边界沿用[文件材料说明](REFERENCE_FILES.md)和[交接说明](HANDOFF_WORKFLOW.md)。

## 配置与用量

默认使用项目 `.env`，可用 `--env-file FILE` 指定其他配置，或 `--no-env` 仅使用进程环境变量。三组 A/B/C 配置与 TUI 相同。有效生成及明确的配置动作才读取模型配置；帮助、材料预览及报告回看均不读取它。

```powershell
.\prompt-optimizer.cmd --configure
.\prompt-optimizer.cmd --config-show --json
.\prompt-optimizer.cmd --config-check --json
.\prompt-optimizer.cmd --config-set run.network_max_attempts 4 --config-set run.prompt_max_repairs 2 --json
```

配置选项专用于配置，不能与生成需求、材料、历史或运行覆盖参数混用，不调用模型。`--configure` 逐项填写并确认保存，密钥使用隐藏输入；回车保留，`/clear` 清空可选值，`/cancel` 或 Ctrl+C 取消。批量 `--config-set KEY VALUE` 可重复，一次完整校验后保存；不接受密钥、重复键或无效值。详细键名、参数来源、生效与回滚行为见 [配置说明](CONFIGURATION.md)。

未显式指定的运行参数读取 `optimizer_settings.json`，缺失项使用内置默认值。`--token-budget N` 覆盖默认 token 调度预算，`--no-token-budget` 清除本次预算。`--network-max-attempts N` 包含首次请求；兼容的 `--retries N` 表示首次之外的重试数，接受非负整数，不再有 5 的上限。这两项互斥。`--handoff-max-attempts`、`--schema-max-attempts`、`--prompt-max-repairs` 分别覆盖交接、结构判断和每会话修复额度；`--no-repair` 继续关闭提示词和交接自动修复。沿用模型决定是否继续澄清的规则，不增加固定澄清轮数、累计请求次数或模型等待时间上限。

信息足够、无交接且未重试的常见流程为一次需求判断、A/B 两次生成和一次 C 评审，共四次模型调用；自动修复另加生成器修复及 C 复核两次。持续澄清、历史整理、格式修复或重试会增加调用，实际用量以报告为准。
