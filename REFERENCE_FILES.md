# 本地文件材料：加载、分块与来源保留

当前产品提供 CLI 与 TUI，双击 `启动提示词优化器.cmd` 直接进入界面；命令行材料参数见 [CLI 使用说明](CLI_WORKFLOW.md)。本地 UTF-8 文本、Markdown、代码或可提取文字的 PDF 可以明确添加为材料。程序保存文件快照，转换为 LangChain `Document` 并分块，生成、修复和评审共用选定内容。

## 在界面中添加

1. 在侧栏材料区输入文件路径，添加到材料列表。窗口较窄时按 F3 展开。
2. 填写材料用途和适用范围，例如“只参考章节结构，事实按本次需求”。材料中的命令和案例不会自动扩大授权。
3. 默认完整加载；确需选块时改为相关片段模式。使用“预览材料”检查来源、位置及覆盖情况，无需模型配置。
4. 在主输入区填写当前需求，F2 或发送提交。历史交接记录使用独立历史区，不能用参考材料代替。

普通路径、网址和自然媒体引用不会触发自动读取，只加载明确添加的本地文件。当前待处理代码或数据属于情境层，借鉴的实现、模板或执行记录属于参考层；必须遵守的要求在指令层明确保留。参考无需输入/输出示例。

## 完整模式与相关片段

`full` 默认保留全部可提取文字和分块重叠，不设选定材料的固定字符上限，不静默摘要或裁剪。`relevant` 按英文标识符和中文相邻双字片段做本地关键词匹配，保留完整块并按原顺序展示，明确标记“未覆盖全文”。每份材料都必须匹配至少一块，否则停止。

相关片段不使用 Embedding、向量数据库或额外模型调用，不保证召回全部相关要求。继续补充时，从缓存的完整块按完整已确认需求重新选择，旧材料快照仍保留。

底层 `ReferenceOptions` 默认块大小 1600、目标重叠 160、相关模式块上限 6；完整模式不使用选块数量限制。`ReferenceOptions.max_chars` 与 `RunOptions.max_input_chars` 默认为 `None`，取消原来的材料 20000 字符和需求与材料合计 50000 字符限制；对话回答及合并后的提示词也不再按固定字符数拦截。程序接口仍可显式传入正整数启用字符预算，超限报错而不截断。模型的上下文窗口、输出 token、会话预算及下面的文件加载规模限制继续生效；字符数不等于 token 数。

## 内容与来源

文本支持 UTF-8/BOM，规范化 CRLF/CR 为 LF，保留代码缩进、变量、其他空白和具体内容。原始字节的 SHA256 保留在每份参考中。记录绝对来源路径、块 ID、起止字符偏移和行号；字符区间从 0 开始，结束位置不包含在区间内，行号从 1 开始。

PDF 按页抽取文字，保留从 1 开始的页码；其中行号和字符偏移对应**该页的提取文本**，不代表视觉行或坐标。空白页或没有文字层的页会列出并提示；整份没有可提取文字则停止。加密、损坏或超限 PDF 会报错。扫描件需要先做 OCR，此入口不自动识别图片、恢复表格版式或读取 PDF 内附件。

生成器只整理任务要求及参考的使用方式。程序把选定快照块原样追加到 A/B 候选和原文候选，再进行独立评审；修复沿用同一份块，已有完整块不重复追加。这样无需让模型重新抄写长代码或文件。输出与报告不能覆盖参考源文件，报告保留 SHA256、覆盖范围、块位置及选定内容。

参考块中的代码与操作仍是材料。生成器与评审共用用途/范围/权限政策，本地保证选定内容的准确定位与保留；这些检查不能证明真实模型始终正确理解参考语义。默认关闭自动 LangSmith 云追踪，正常优化时选定参考会进入你配置的模型请求。离线预览完全在本机处理。

## 接入与边界

实现使用 LangChain `BaseLoader` 接口、`Document` 和官方 `RecursiveCharacterTextSplitter`。本项目的 `LocalReferenceLoader` 从同一份有大小限制的字节快照读取文本，或调用 `pypdf` 解析 PDF；没有依赖 `langchain-community` 的加载器集合。

Markdown 和已支持的代码语言使用 LangChain 的语言分隔规则，普通文本补入中文标点分隔；每块回查原文位置，并核对分块覆盖所有提取字符。分隔规则优先按结构拆分，但没有 AST 保证：超长函数、代码围栏或章节仍可能被拆开。对应接口及规则见 [LangChain 文件加载说明](https://docs.langchain.com/oss/python/integrations/document_loaders)、[递归分块说明](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)和[代码分块说明](https://docs.langchain.com/oss/python/integrations/splitters/code_splitter)。PDF 文字层及 OCR 限制见 [pypdf 官方说明](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)。

新增项目依赖 `langchain-text-splitters==1.1.3`、`pypdf==6.19.0`，已有 LangChain Core、OpenAI 适配器及 OpenEvals 版本保持不变。没有接入模型结果缓存、LangGraph 持久化、网络链接加载或整个目录的自动扫描。

底层默认限制每次 10 个文件、单文件 5 MiB、单文件提取文本 200000 字符、PDF 200 页；程序接口可以明确调整。这些是输入规模限制，不是针对任意恶意文件的隔离环境。`.env` 及其变体、指向配置的路径不作为参考或任务正文读取；模型运行时的配置加载方式沿用原流程。

```python
from pathlib import Path
from optimizer_documents import ReferenceFile, ReferenceOptions, prepare_references
from optimizer_engine import Optimizer, RunOptions

references = prepare_references([
    ReferenceFile(Path("reference.py"), purpose="参考接口结构", usage="不复制业务事实"),
    ReferenceFile(Path("report.md"), purpose="参考报告章节", usage="按本次材料填写"),
], ReferenceOptions(mode="full"))
# configs 沿用原有显式配置；此处只展示接入方式。
optimizer = Optimizer(configs, RunOptions(), references=references)
# result = await optimizer.run(original_request)
```

## 历史验证与当前回归

以下为文件材料功能首次实现时的验收证据；模式名称与测试数量对应当时版本。当前产品固定 A/B 与独立评审，最新回归见 [TUI 验收记录](TUI_ACCEPTANCE.md)。

验证使用真实 LangChain 分块器、`pypdf` 生成/解析的临时 PDF、本机 HTTP 模拟接口和替代模型，覆盖 UTF-8/BOM、中文和代码空白、页码、完整覆盖、相关片段、预算、配置隔离、源文件保护、同一快照、快速/质量/修复。旧菜单测试仅使用历史夹具；当前界面与入口另由 TUI 测试覆盖。

全量 **197 项测试通过**（11.605 秒），其中新增 30 项；`pip check` 返回 `No broken requirements found.`。实际运行演示模板离线预览，得到 1 份文件、1/1 块、404 字符，完整覆盖提取文本；记录见 离线预览结果（仅本地保留：`evals/results/20261003-reference-files-offline.json`）。

没有读取真实 `.env`，没有调用外部模型；因此不能据此宣称优化质量或检索召回提升。回归入口：

```powershell
& .\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -q
& .\.venv\Scripts\python.exe -m pip check
```

实现：[optimizer_documents.py](optimizer_documents.py)；检查：[test_documents.py](tests/test_documents.py)。
