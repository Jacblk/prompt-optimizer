# 统一配置说明

独立配置程序、TUI 设置入口、交互 CLI 和批量 CLI 共用 `optimizer_settings.py` 的参数定义、校验、窗口绑定和变更提交服务。配置操作只在本地进行，不创建模型客户端，不调用模型。

## 入口

- 双击 `配置提示词优化器.cmd`，或运行项目 Python 的 `configure.py`，打开独立表单。
- TUI 设置页点击“配置模型与参数”，打开同一表单。80×24 窄屏可滚动，常用和高级参数分别成页；Esc 取消。
- `prompt-optimizer.cmd --configure` 是逐项配置向导，密钥使用隐藏输入。回车保留已有值，`/clear` 清空可选值，`/cancel` 或 Ctrl+C 取消；保存前展示改动键名并确认。
- `--config-show` 查看有效参数、来源和密钥是否配置；密钥值和额外请求参数的内容不展示。
- `--config-check` 离线检查模型缺失项、窗口绑定和输出容量，分别标出 `generation_ready` 与 `handoff_ready`。普通生成配置可就绪而交接窗口尚未就绪；全部有效返回 0，否则返回 2。
- `--config-set KEY VALUE` 可重复，一批完整校验后保存。未知键、重复键、密钥参数或无效值均不写入。

```powershell
.\prompt-optimizer.cmd --configure
.\prompt-optimizer.cmd --config-show --json
.\prompt-optimizer.cmd --config-check --json
.\prompt-optimizer.cmd --config-set run.network_max_attempts 4 --config-set run.prompt_max_repairs 2 --json
.\prompt-optimizer.cmd --config-set models.a.max_tokens 200000 --config-set models.a.context_window 500000
.\prompt-optimizer.cmd --config-set run.token_budget null
# 旧窗口命令继续调用相同的窗口服务；数字仅示意顺序
.\prompt-optimizer.cmd --configure-contexts 128000 128000 128000
```

配置动作不能与生成需求、历史、材料或运行覆盖参数混用。它们支持 `--json`；查看、检查和保存也默认输出可读 JSON，向导提示写入 stderr。不要通过普通命令参数提供密钥，使用向导或表单的隐藏输入。密钥留空表示保留，不能用空值删除。

## 参数与默认值

CLI 键名中的 A/B/C 为 `models.a`、`models.b`、`models.c`，兼容 `models.judge` 作为 C 的别名。C 在模型服务配置中仍使用 `JUDGE` 前缀，A/B 使用 `GENERATOR_A`／`GENERATOR_B`。

### 模型参数

下表每个普通字段均可加在 `models.a.`、`models.b.` 或 `models.c.` 后。

| 字段 | 默认值／取值 | 作用 |
| --- | --- | --- |
| `name` | 必填 | 服务提供的模型名称 |
| `base_url` | 必填 | HTTP(S) Chat Completions API 基地址，不含凭据、查询参数或片段 |
| `api_key` | 必填；隐藏输入 | 只在表单或向导中编辑，留空保留 |
| `max_tokens` | 8192；正整数 | 普通生成及最终评审输出上限，无应用固定上限 |
| `context_window` | 未配置；正整数 | 服务实际上下文窗口，保存在窗口文件 |
| `temperature` | 空；0 到 2 | 空时不发送温度字段 |
| `reasoning_effort` | 空；none/minimal/low/medium/high/xhigh/max | 设置 `extra_body.reasoning_effort`；空时移除该字段 |
| `timeout` | 90 秒；有限数值且 ≥0.1 | 连接、发送与连接池超时，不限制响应读取等待 |
| `slow_warning_seconds` | 90 秒；有限数值且 ≥0.1 | 无有效活动提醒，只提醒不停止 |
| `token_limit_field` | max_tokens | 可选 max_tokens 或 max_completion_tokens |
| `json_mode` | false | true/false，按服务支持设置 JSON 输出模式 |
| `extra_body` | `{}` | 服务专有 JSON 对象；禁止重复键、非有限数字及覆盖受保护请求字段 |

A/C 另有 `handoff_max_tokens` 与 `handoff_reasoning_effort`，默认留空。交接输出额度在“常用”页修改，交接推理强度在“高级”页修改。显式交接输出值优先；留空时，思考模式的 `deepseek-flash`、`deepseek-v4-pro` 在 A/C 的交接额度为普通额度与 65536 的较大值，其他模型沿用普通额度。交接推理强度留空沿用普通配置。底部按 A/B/C 三行预览各阶段额度：A 显示提示词生成与历史整理，B 显示普通／交接共用的提示词生成，C 显示候选评审与历史核验。B 不参与历史整理，没有独立的历史输出额度；修改 B 的普通输出上限即可调整它在两种流程中的生成额度。`--config-show` 提供 `effective_handoff_max_tokens`；预检和实际请求使用相同额度。服务自身的限制仍然适用。

### 尝试额度与运行参数

| 键 | 默认值 | 范围与计数方式 |
| --- | --- | --- |
| `run.network_max_attempts` | 2 | 正整数，含首次网络请求；每角色每会话另允许一次流式兼容回退 |
| `run.handoff_max_attempts` | 2 | 正整数；每个历史提取、归并或更新阶段分别计算，每次包含整理与独立核验；截断时不核验未完成正文 |
| `run.schema_max_attempts` | 2 | 正整数；每次澄清或底层分层结构判断的结构纠正次数，含首次 |
| `run.prompt_max_repairs` | 1 | 非负整数，每会话累计；0 关闭，每次修复后完整 C 评审，额度耗尽明确停止 |
| `run.token_budget` | null | 可选正整数；null 不设累计 token 调度预算 |
| `run.max_input_chars` | null | 可选正整数；null 不设应用字符上限 |
| `run.max_builtin_examples` | 3 | 非负整数；0 不加入内置示例 |
| `run.max_example_chars` | 4000 | 非负整数；内置示例字符预算 |

网络尝试和输出 token 不再受旧的 5 次重试、131072 token 固定上限约束。不会新增累计会话请求次数或时间上限。网络失败、截断、结构纠正、整理核验和修复实际请求均进入既有账本；可选 token 预算与未知用量停止规则继续适用。

### 历史与材料参数

| 键 | 默认值 | 范围／单位 |
| --- | --- | --- |
| `history.chunk_bytes` | 24000 | 正整数，UTF-8 字节分块；窗口预检仍可缩小分块 |
| `history.max_files` | 10 | 1 到 10 |
| `history.max_file_bytes` | 5242880 | 1 到 5242880 字节（5 MiB） |
| `history.max_chars` | 200000 | 1 到 200000，总历史字符数 |
| `reference.mode` | full | full 全文；relevant 关键词相关片段 |
| `reference.chunk_size` | 1600 | 正整数，字符 |
| `reference.chunk_overlap` | 160 | 非负整数，必须小于 chunk_size |
| `reference.max_chunks` | 6 | 正整数，相关片段数量 |
| `reference.max_chars` | null | 可选正整数，选中材料字符预算；null 不限制 |
| `reference.max_files` | 10 | 1 到 10 |
| `reference.max_file_bytes` | 5242880 | 1 到 5242880 字节 |
| `reference.max_extracted_chars` | 200000 | 1 到 200000，单文件提取字符数 |
| `reference.max_pdf_pages` | 200 | 1 到 200 |

文件规模、PDF 和历史读取限制只能在当前安全边界内调低，不能通过配置解除。格式、二进制内容、材料权限与实际请求容量检查继续执行。

## 保存、优先级与生效

项目 `.env` 保存模型参数与密钥，`optimizer_settings.json` 保存版本为 1 的运行／历史／材料默认值，`context_windows.json` 只保存本地模型／服务窗口绑定。三个配置文件均不随仓库发布；公开的 `context_windows.example.json` 只有版本与空角色表。首次通过配置程序或 `--configure-contexts` 保存有效窗口时创建本地 `context_windows.json`；已有文件继续使用。窗口模板受到配置与生成保存保护。没有运行参数文件时使用内置默认值，可参考 `optimizer_settings.example.json`。表单仅展示当前流程实际读取的字段；原 `.env` 中的未知键、未使用的 `MODEL_*` 兼容键和注释保留，不复制到新表单。开发专用模式开关和已经移除的轮数、时间、请求上限不重新开放。

模型值优先级为进程环境覆盖 `.env`；`--config-show` 与表单标出来源。表单编辑文件值，有进程覆盖时会提示保存不能改变该进程环境。运行参数优先级为 CLI 显式选项／TUI 本次会话设置、保存默认值、内置默认值。CLI `--retries N` 继续表示首次之外的重试数，与 `--network-max-attempts` 互斥；`--no-repair` 继续关闭提示词与交接自动修复，`--no-token-budget` 可清除保存的默认预算。

整批变更先校验，全部通过才逐文件原子替换；保留无关模型配置和注释。保存失败尝试恢复原文件的完整字节，报告是否回滚成功；无法回滚时列出仍需核对的文件。多文件提交不是文件系统级全局事务，期间并发修改会被检测并拒绝。取消或仅核验不保存，错误和变更摘要不包含密钥。损坏的运行 JSON 在普通查看／检查／批量修改时拒绝；明确打开配置表单或向导可恢复默认值，展示恢复说明，只有保存才替换原文件。

更换模型名称或服务地址会清除该角色的旧窗口绑定，其他角色的有效绑定保留。随后重新填写窗口，或在 CLI 同一批变更中明确提交新身份与窗口值。配置检查只能检查输出预留和绑定，具体需求／历史的实际容量仍在提交及每次调用边界检查。

模型和运行参数从新会话生效；已有会话保持原值。TUI 内保存后提示 Ctrl+N，因配置变更新建会话会保留未提交需求、材料、历史和上次显示结果；独立配置程序或 CLI 修改后也通过文件元数据识别，启动界面不因此读取密钥。上次结果仅供回看。窗口有效则直接发送，缺失、绑定失效或不足时停止本次提交，提供配置入口，不弹旧窗口表单、不自动重发；配置好后由用户再次发送。

`--env-file`、`--no-env`、`--context-config` 保留。`--no-env` 可查看／检查进程模型配置并保存运行默认值或窗口，不能把模型变更写入不存在的模型文件。所有配置目标受到源文件、示例、成功结果与链接保护。运行参数文件加入生成保存路径保护和发布排除，不作为需求、材料、历史或报告读取；密钥不进入生成报告。

## 离线验证

`tests/test_configuration.py`、`tests/test_tui.py`、`tests/test_tui_integration.py` 覆盖共享读取、批量保存、隐藏密钥、回滚、取消、来源、会话切换和 80×24 交互。`tests/test_adapter.py` 用安装的 SDK 与本机回环服务核对超出旧上限的普通与交接参数；`tests/test_model_activity.py` 核对网络尝试与独立流式回退计数，发布测试确认本地参数不被读取、哈希或导出。最终结果见 [验收记录](TUI_ACCEPTANCE.md)。真实模型验证继续暂不执行。
