"""Hand-authored few-shot demonstrations, separate from evaluation inputs."""

from optimizer_layers import ArtifactReference, ImplementationReference, InputOutputReference, render_reference


def _reference_block(materials):
    return "参考材料（模型补充，仅作演示）：\n" + "\n\n".join(
        render_reference(material, index) for index, material in enumerate(materials, 1))


_DEVICE_EXAMPLES = _reference_block([
    InputOutputReference(kind="input_output", purpose="演示状态与 JSON 输出的对应关系。",
                     usage="参考 state 字段和标签；演示记录不是当前输入。",
                     input_text="演示：设备持续响应心跳。", output_text='{"state":"在线"}'),
    InputOutputReference(kind="input_output", purpose="演示状态与 JSON 输出的对应关系。",
                     usage="参考 state 字段和标签；演示记录不是当前输入。",
                     input_text="演示：设备断电，无心跳。", output_text='{"state":"离线"}'),
])
_CODE_REFERENCE = _reference_block([
    ImplementationReference(kind="implementation", purpose="演示空列表处理的最小代码改动。",
        usage="仅借鉴边界处理方法；适配 {source_code}，不引入演示函数或新文件。",
        content="演示代码：\ndef first(items):\n    return items[0] if items else None"),
])
_FILE_REFERENCE = _reference_block([
    ArtifactReference(kind="artifact", purpose="演示 Markdown 报告的章节和可核对依据。",
        usage="仅参考排版；内容依据 {notes} 生成，演示章节不作为新增必填要求。",
        content="# 报告标题（演示）\n\n## 发现\n[材料支持的发现]\n\n## 依据\n[对应材料位置]"),
])

# These teach how to rewrite a request, not how to answer its underlying task.
GENERATION_EXAMPLES = [
    {
        "original_request": "缓存是什么给我讲讲",
        "output": {
            "status": "ready",
            "optimized_prompt": "请解释缓存的基本概念与用途。",
            "preserved_constraints": [],
            "clarification_questions": [],
            "change_summary": ["把口语化请求整理为明确的解释任务，未指定原文没有的受众或篇幅。"],
        },
    },
    {
        "original_request": "给刚上岗的馆员讲解我随后提供的借阅登记流程，亲切但不随意，先概括用途再说明步骤，只以我给的流程为准，200 字以内。",
        "output": {
            "status": "ready",
            "optimized_prompt": (
                "## 指令层\n"
                "请讲解借阅登记流程，内容仅以提供的流程为准。\n"
                "1. 概括流程用途。\n"
                "2. 说明办理步骤。\n\n"
                "## 情境层\n"
                "待讲解材料：随后提供的借阅登记流程。\n\n"
                "## 输出层\n"
                "面向刚上岗的馆员，亲切但不随意，全文不超过 200 字。"
            ),
            "preserved_constraints": [
                {"source_quote": "给刚上岗的馆员", "constraint": "受众是刚上岗的馆员。"},
                {"source_quote": "我随后提供的借阅登记流程", "constraint": "保留稍后提供的输入材料，不在优化阶段索要或虚构流程。"},
                {"source_quote": "亲切但不随意", "constraint": "保留语气的两方面要求。"},
                {"source_quote": "先概括用途再说明步骤", "constraint": "保持先用途、后步骤的顺序。"},
                {"source_quote": "只以我给的流程为准，200 字以内", "constraint": "保留内容来源与篇幅限制。"},
            ],
            "clarification_questions": [],
            "change_summary": ["分开任务、表达和内容边界，将已有步骤按顺序编号并逐行组织。"],
        },
    },
    {
        "original_request": "这是反复用的模板。照例子给状态文本打标签，正常启动 → OK；启动失败 → ERROR。文本是 {status_text}。只输出标签。",
        "output": {
            "status": "ready",
            "optimized_prompt": (
                "## 指令层\n请按以下参考映射，为状态文本选择标签。\n\n"
                "## 情境层\n这是可复用模板，待判断文本：{status_text}\n\n"
                "## 参考层\n示例输入：正常启动\n示例输出：OK\n"
                "示例输入：启动失败\n示例输出：ERROR\n\n"
                "## 输出层\n只输出标签。"
            ),
            "preserved_constraints": [
                {"source_quote": "这是反复用的模板", "constraint": "保持模板可复用，不代填变量。"},
                {"source_quote": "正常启动 → OK；启动失败 → ERROR", "constraint": "保留示例原文与标签对应关系。"},
                {"source_quote": "{status_text}", "constraint": "变量名称及花括号保持不变。"},
                {"source_quote": "只输出标签", "constraint": "下游输出不增加解释或分析过程。"},
            ],
            "clarification_questions": [],
            "change_summary": ["把已有示例整理为输入/输出对，保留模板变量和标签输出约定。"],
        },
    },
    {
        "original_request": '这是可复用模板：将巡检记录中的设备状态标为在线或离线，只输出 {"state":"在线或离线"} 这样的 JSON，不要其他字段。待处理记录：{record}。',
        "layer_decisions": [{"layer": "references", "choice": "model", "value": _DEVICE_EXAMPLES}],
        "output": {
            "status": "ready",
            "optimized_prompt": (
                "## 指令层\n请根据巡检记录判断设备是在线还是离线。\n\n"
                "## 情境层\n这是可复用模板，待处理记录：{record}\n\n"
                "## 参考层\n" + _DEVICE_EXAMPLES + "\n\n"
                '## 输出层\n仅输出 {"state":"在线或离线"} 这样的 JSON，state 为对应状态，不要其他字段。'
            ),
            "preserved_constraints": [
                {"source_quote": "在线或离线", "constraint": "保留用户给定的两个状态。"},
                {"source_quote": '只输出 {"state":"在线或离线"} 这样的 JSON，不要其他字段',
                 "constraint": "示例输出及下游结果都只含 state 字段，不增加分析字段。"},
                {"source_quote": "{record}", "constraint": "变量保留为新输入，与示例分开。"},
                {"source_quote": _DEVICE_EXAMPLES, "constraint": "完整保留用户采用的模型示例块及输入/输出映射。"},
            ],
            "clarification_questions": [],
            "change_summary": ["区分情境层的当前输入与参考层的已选演示，原样保留映射及 JSON 输出约定。"],
        },
    },
    {
        "original_request": "根据{方案甲}和{方案乙}帮现场维护人员选一个备份方案，只比较恢复时间和日常维护工作量。先比较再推荐，说明依据；不要讨论采购价格，也别编造测试结果。",
        "output": {
            "status": "ready",
            "optimized_prompt": (
                "## 指令层\n请比较两个备份方案并推荐一个。\n"
                "只比较恢复时间和日常维护工作量，不讨论采购价格，不编造测试结果；材料无法支持的判断明确说明依据不足。\n"
                "1. 从材料中提取两个方案在这两个维度上的依据。\n"
                "2. 按同一组维度逐项比较和权衡。\n"
                "3. 给出推荐及简短依据。\n"
                "4. 核对推荐是否符合原条件。\n\n"
                "## 情境层\n待比较材料：{方案甲}和{方案乙}。\n\n"
                "## 输出层\n面向现场维护人员，先给比较，再给推荐及依据。"
            ),
            "preserved_constraints": [
                {"source_quote": "{方案甲}和{方案乙}", "constraint": "保留两个方案变量，不代填或预选赢家。"},
                {"source_quote": "现场维护人员", "constraint": "维持原有受众。"},
                {"source_quote": "只比较恢复时间和日常维护工作量", "constraint": "不扩展比较维度。"},
                {"source_quote": "先比较再推荐，说明依据", "constraint": "保留比较到推荐的顺序，要求简短可核对的依据。"},
                {"source_quote": "不要讨论采购价格，也别编造测试结果", "constraint": "保留排除项与事实边界。"},
            ],
            "clarification_questions": [],
            "change_summary": ["加入提取依据、逐项比较与权衡、形成推荐后复核的 CoT 引导，保留范围和顺序，不执行方案选择。"],
        },
    },
    {
        "original_request": "给告警消息打标签，错误必须标 ERROR，正常必须标 OK。参考示例：磁盘读取错误 → OK。待分类内容：{alert}。只输出标签。",
        "output": {
            "status": "needs_clarification",
            "optimized_prompt": (
                "## 指令层\n请为告警消息选择标签；错误必须标 ERROR，正常必须标 OK。\n"
                "[待确认：错误标记规则与参考示例冲突，须确认采用哪一种约定。]\n\n"
                "## 情境层\n待分类内容：{alert}\n\n"
                "## 参考层\n参考示例：磁盘读取错误 → OK。\n\n"
                "## 输出层\n最终只输出标签。"
            ),
            "preserved_constraints": [
                {"source_quote": "错误必须标 ERROR，正常必须标 OK", "constraint": "保留明确的标签规则。"},
                {"source_quote": "磁盘读取错误 → OK", "constraint": "保留冲突示例以供确认，不擅改成 ERROR。"},
                {"source_quote": "{alert}", "constraint": "保留输入变量，不实际分类。"},
                {"source_quote": "只输出标签", "constraint": "保留下游输出约定。"},
            ],
            "clarification_questions": ["“错误必须标 ERROR”与“磁盘读取错误 → OK”冲突，应采用哪一种约定？"],
            "change_summary": ["整理标签任务并指出原有规则与示例冲突，保留双方等待确认。"],
        },
    },
    {
        "original_request": "帮实习同事分析 {run_logs} 里的离线转码变慢问题。逐步排查，说明每步依据；只给检查建议，不改配置、不删除缓存、不运行命令。",
        "output": {
            "status": "ready",
            "optimized_prompt": (
                "## 指令层\n请整理离线转码变慢的排查建议。\n"
                "只给检查建议，不改配置、不删除缓存、不运行命令。\n"
                "1. 梳理已知现象，区分日志事实与待验证假设。\n"
                "2. 按定位依赖逐步安排检查，说明各项检查的对象、目的及哪些证据支持或排除假设；结果会影响后续检查时说明对应判断分支。\n"
                "3. 根据检查证据形成结论和后续建议，证据不足时保留不确定性，不直接认定根因。\n\n"
                "## 情境层\n排查依据：{run_logs}\n\n"
                "## 输出层\n面向实习同事，逐步给出检查建议及每步依据。"
            ),
            "preserved_constraints": [
                {"source_quote": "实习同事", "constraint": "保留原有受众，不新增角色身份。"},
                {"source_quote": "{run_logs}", "constraint": "保留日志变量，不猜测运行环境。"},
                {"source_quote": "离线转码变慢", "constraint": "保持原诊断对象。"},
                {"source_quote": "逐步排查，说明每步依据", "constraint": "引导按证据逐步排查，再得出有条件的结论。"},
                {"source_quote": "只给检查建议，不改配置、不删除缓存、不运行命令", "constraint": "推理和核验不扩大操作权限。"},
            ],
            "clarification_questions": [],
            "change_summary": ["加入现象与假设区分、证据核验、结果分支到条件性结论的 CoT 引导，仅改写排查任务。"],
        },
    },
    {
        "original_request": '根据 {inventory} 中的数量、单价和折扣，判断订单总额 {claimed_total} 是否计算正确。最后只输出 {"valid":true} 这种 JSON，valid 是布尔值，不要其他字段。',
        "output": {
            "status": "ready",
            "optimized_prompt": (
                "## 指令层\n请核对订单总额是否计算正确。\n"
                "1. 逐项核对给定数量、单价及折扣条件。\n"
                "2. 按所给规则计算应付总额并与申报总额对照。\n"
                "3. 复核关键计算和结论。\n\n"
                "## 情境层\n订单材料：{inventory}\n申报总额：{claimed_total}\n\n"
                '## 输出层\n最终只输出 {"valid":true} 这种 JSON，valid 是核验结果的布尔值，不是固定为 true；不要其他字段或 JSON 外的分析。'
            ),
            "preserved_constraints": [
                {"source_quote": "{inventory}", "constraint": "保留输入变量，只使用给定数量、单价与折扣规则。"},
                {"source_quote": "{claimed_total}", "constraint": "保留待核验总额，不在优化阶段计算或代填。"},
                {"source_quote": '最后只输出 {"valid":true} 这种 JSON，valid 是布尔值，不要其他字段',
                 "constraint": "保留单字段 JSON 与布尔值类型，推导与核验不改变最终输出。"},
            ],
            "clarification_questions": [],
            "change_summary": ["加入条件核对、按规则计算、比较与复核的 CoT 引导，最终仍只输出原约定的 JSON。"],
        },
    },
    {
        "original_request": "修改 {source_code} 的空列表处理：空列表返回 None，非空列表保持原行为。直接修改现有文件，不新增文件；最后简短说明改动。",
        "layer_decisions": [{"layer": "references", "choice": "model", "value": _CODE_REFERENCE}],
        "output": {
            "status": "ready",
            "optimized_prompt": (
                "## 指令层\n直接修改现有文件，使空列表返回 None，非空列表保持原行为；不新增文件。\n\n"
                "## 情境层\n待修改代码：{source_code}\n\n"
                "## 参考层\n借鉴边界处理方法并适配实际代码：\n" + _CODE_REFERENCE + "\n\n"
                "## 输出层\n最后简短说明改动。"
            ),
            "preserved_constraints": [
                {"source_quote": "空列表返回 None，非空列表保持原行为", "constraint": "保留边界条件和原行为。"},
                {"source_quote": "直接修改现有文件，不新增文件", "constraint": "交付实际代码改动，不用聊天汇报代替，不新增文件。"},
                {"source_quote": "最后简短说明改动", "constraint": "最终说明保持简短，参考代码不改变交付方式。"},
                {"source_quote": _CODE_REFERENCE, "constraint": "原样保留参考内容和借鉴范围，不复制演示函数。"},
            ],
            "clarification_questions": [],
            "change_summary": ["纳入用户采用的参考实现及适用范围，区分当前代码与演示代码，保留实际修改和汇报要求。"],
        },
    },
    {
        "original_request": "依据 {notes} 生成 Markdown 报告并保存到 {destination}。不要补写材料没有的事实，最后仅给文件路径。",
        "layer_decisions": [{"layer": "references", "choice": "model", "value": _FILE_REFERENCE}],
        "output": {
            "status": "ready",
            "optimized_prompt": (
                "## 指令层\n依据材料生成 Markdown 报告，不补写材料没有的事实。\n\n"
                "## 情境层\n内容依据：{notes}\n\n"
                "## 参考层\n下面的文件样本仅供排版参考：\n" + _FILE_REFERENCE + "\n\n"
                "## 输出层\n实际保存报告到 {destination}，最后仅给文件路径。"
            ),
            "preserved_constraints": [
                {"source_quote": "{notes}", "constraint": "内容只依据实际材料，不采用演示占位内容。"},
                {"source_quote": "生成 Markdown 报告并保存到 {destination}", "constraint": "保留格式、实际文件交付及目标路径变量。"},
                {"source_quote": "不要补写材料没有的事实", "constraint": "不补造事实。"},
                {"source_quote": "最后仅给文件路径", "constraint": "最终回复只给路径，不替代文件交付。"},
                {"source_quote": _FILE_REFERENCE, "constraint": "原样保留样本、排版用途与适用范围，不把演示章节升级为硬要求。"},
            ],
            "clarification_questions": [],
            "change_summary": ["加入文件内容参考，区分文件交付与最终回复，不强制套用输入/输出问答。"],
        },
    },
    {
        "original_request": "根据所给视频，整理维修操作的先后顺序，只列操作步骤。",
        "output": {
            "status": "ready",
            "optimized_prompt": "请根据所给视频，按操作发生的先后顺序整理维修步骤。只列操作步骤。",
            "preserved_constraints": [
                {"source_quote": "所给视频", "constraint": "保留待处理视频的自然引用，不猜测视频内容或改为字幕输入。"},
                {"source_quote": "维修操作的先后顺序", "constraint": "保留操作对象及顺序要求。"},
                {"source_quote": "只列操作步骤", "constraint": "保留最终输出范围，不增加说明或其他字段。"},
            ],
            "clarification_questions": [],
            "change_summary": ["整理任务与顺序要求，保留自然视频引用；不因素材不可见而要求补全。"],
        },
    },
    {
        "original_request": "按照附件来。",
        "output": {
            "status": "needs_clarification",
            "optimized_prompt": "请按照附件完成[待确认：任务目标及附件用途]。",
            "preserved_constraints": [
                {"source_quote": "附件", "constraint": "保留附件引用，不用虚构材料替代。"},
            ],
            "clarification_questions": ["希望完成什么任务，附件用于提供待处理内容、参考样式，还是必须遵循的标准？"],
            "change_summary": ["保留附件绑定，仅确认未说明的任务目标与用途，不猜测附件内容。"],
        },
    },
]
