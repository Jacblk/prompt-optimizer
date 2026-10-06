# 官方评审组件接入

> 资料状态（2026-10-04）：本文保留历史分析与结论，课名和时间点可用于回看原视频。课程 OCR、字幕、截图、脚本及来源清单已清理；见 [保留资料说明](../README.md)。当前功能与操作以主 README 为准。

2026-10-02。继续使用 LangChain 调用模型；质量模式的评审入口改为官方 OpenEvals `create_async_llm_as_judge`，固定发布版本 `0.2.0`。上游主分支当时标为 `0.2.1`，PyPI 尚未发布该版本，安装采用实际可取得的版本。

## 流程与职责

```text
A / B 并行生成
  → 原文与匿名候选
  → OpenEvals（原有 review_prompt + Review 输出结构）
  → LangChain 预算适配器
  → Optimizer._call(judge) → 现有 ChatOpenAI / Chat Completions 接口
  → 严格 JSON / Pydantic 校验
  → 候选完整性、引用、权限与动作校验
  → 选定、澄清、需复核，或一次修复及重新评审
```

`optimizer_review.py` 提供官方评审与现有请求管理之间的适配。OpenEvals 使用自定义 `Review`，一次返回所有候选的判断、问题证据、评分及下一步动作。没有把逐条指标拆成多次收费请求；A、B、评审仍是三个配置角色。

适配器是 LangChain `BaseChatModel`，通过原有 `_call` 发起请求，而不是另建客户端。请求数、重试、时间、token、未知用量和取消继续由引擎管理。`with_structured_output` 在本地严格解析响应，以兼容只支持普通 Chat Completions 或 JSON Object 的服务；不强制增加工具调用或 JSON Schema 请求参数。

评审提示通过可调用消息构造器传入，保留字面 JSON、花括号变量、中文和换行。OpenEvals 及底层 SDK 均置于禁用自动云追踪的上下文中。质量模式在生成前检查依赖；快速模式不调用 OpenEvals。报告新增 `review_backend` 名称与版本，便于辨认实际评审入口。

## 与 CoT 的关系

初始接入沿用 `2.4.0` 的生成、修复与评审规则：根据比较、排障、计算、逻辑和规划任务安排有依据的推导与核验，保留步数、权限和最终展示限制。

真实检查发现旧评审把“不要添加思维链或分步处理要求”当成可删除元说明。随后将共享规则补强为：此类明确排除须在优化稿中显式保留，可等价重述；只输出标签/答案不能替代推理方法排除。提示版本升至 `2.4.1`。对同一批真实候选再做一次真实评审后，删除排除条款的候选被判失败，最终保留原文。

官方组件负责评审执行和结构输出；原意、范围、约束、事实及 CoT 方法的判据仍由项目定义。复用官方组件不会自动提高判断准确率，本次主要验证接口与流程兼容性。

## 依赖与可复查材料

- 固定 `openevals==0.2.0`、已有 `langchain==1.4.3`、新增 `rich==15.0.0`；原有核心 SDK 固定版本保持一致。
- 安装预检只新增 OpenEvals、Rich 及 Rich 的三个间接依赖，没有替换现有 SDK；安装后 `pip check` 通过。
- 原引擎、模型适配器和依赖清单归档于 `baselines/before_openevals_20261002/`（仅本地保留）。
- `evals/artifacts/openevals-install-plan.json`（仅本地保留）：公开 PyPI 依赖预检。
- `evals/artifacts/openevals-offline-tests.txt`（仅本地保留）：132 项离线测试全部通过，9.683 秒。覆盖实际官方工厂调用、本地 SDK 接口、严格解析、取消、禁用追踪，以及既有工作流回归。
- `evals/review_probe.py`：默认只列计划；`--run --load-env` 才进行真实检查。三组最多 10 次请求，80,000 token 软调度阈值，关闭重试和修复。程序仅加载三个质量模式配置，不展示或记录配置值。结果逐次写入 `evals/results/`（仅本地保留）。

初轮第一组通过，第二组评审在 90 秒时超时，记录为未知用量并停止后续请求。保留原报告后，对尚未完成的两组独立追加最多 7 次请求，仅通过 `--judge-timeout 180 --workflow-timeout 450` 放宽本次测试等待时间，没有修改配置文件。已知 token 和未知用量分开报告，超时请求的费用不按零计算。

另用 `evals/review_policy_probe.py` 对缓存候选做一次新判据复核，不重新生成。三轮共 14 次请求，13 次成功、1 次超时，已知用量 77,708 tokens，另有一笔未知。原始记录、语义复查、复现方式与限制见完整检查记录（仅本地保留：`evals/results/20261002-openevals-review.md`）。

当前 OpenEvals / LangSmith 在 Python 3.14 下会报告 `asyncio.iscoroutinefunction` 的弃用警告；测试中的调用成功，尚不能据此保证未来 Python 3.16 兼容。

## 官方依据

[OpenEvals 官方说明](https://github.com/langchain-ai/openevals#llm-as-judge)介绍异步评审、LangChain 模型和自定义输出结构。本次也核对了虚拟环境中实际安装的 `openevals/llm.py`、`utils.py`，并由离线测试确认所安装版本的行为。
