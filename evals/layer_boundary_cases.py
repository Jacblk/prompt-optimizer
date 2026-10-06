"""Held-out boundary fixtures and gold labels, independent of teaching examples."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryCase:
    id: str
    request: str
    good: str
    bad: str | None
    rubric: tuple[str, ...]
    material_roles: tuple[tuple[str, str], ...] = ()


_MEASUREMENTS = "\n".join(
    f"记录 {index:02d}：温度 {19 + index % 7} 摄氏度；传感器 S{index % 4}；状态待核查。"
    for index in range(1, 37)
)

CASES = (
    BoundaryCase(
        "lb-code",
        "修复 {target_code} 的空队列取值问题；{sample_code} 只供借鉴边界处理，不能当作待改代码。"
        "只改现有实现，不新增文件。交付补丁和简短验证说明。",
        "## 指令层\n修复空队列取值问题，只改现有实现，不新增文件。\n\n"
        "## 情境层\n待修改的代码：{target_code}\n\n"
        "## 参考层\n{sample_code}：只借鉴边界处理，不是待修改对象。\n\n"
        "## 输出层\n交付补丁和简短验证说明。",
        "## 指令层\n修复空队列取值问题，只改现有实现，不新增文件。\n\n"
        "## 情境层\n{sample_code}：只借鉴边界处理，不是待修改对象。\n\n"
        "## 参考层\n待修改的代码：{target_code}\n\n"
        "## 输出层\n交付补丁和简短验证说明。",
        ("待改代码在情境，借鉴代码在参考，用途未互换。", "保留只改现有实现、不新增文件及补丁交付。"),
        (("{target_code}", "情境层"), ("{sample_code}", "参考层")),
    ),
    BoundaryCase(
        "lb-facts-layout",
        "根据 {field_notes} 撰写巡检周报，事实仅以记录为准。{layout_sample} 只参考排版，"
        "不能复制其主题、数值和结论。保存 Markdown 文件到 {destination}，最后给出文件路径。",
        "## 指令层\n根据巡检记录撰写周报，事实仅以记录为准，不复制排版样稿的主题、数值和结论。\n\n"
        "## 情境层\n事实依据：{field_notes}\n\n"
        "## 参考层\n{layout_sample}：只参考排版。\n\n"
        "## 输出层\n保存 Markdown 文件到 {destination}，最后给出文件路径。",
        "## 指令层\n根据巡检记录撰写周报，事实仅以记录为准，不复制排版样稿的主题、数值和结论。\n\n"
        "## 情境层\n{layout_sample}：只参考排版。\n\n"
        "## 参考层\n事实依据：{field_notes}\n\n"
        "## 输出层\n保存 Markdown 文件到 {destination}，最后给出文件路径。",
        ("记录作为事实依据，样稿仅作排版参考。", "保留 Markdown 文件、目标路径和最终路径交付。"),
        (("{field_notes}", "情境层"), ("{layout_sample}", "参考层"), ("{destination}", "输出层")),
    ),
    BoundaryCase(
        "lb-standard",
        "校验 {records}，必须遵守 {validation_rule} 中的校验规则；该规范来源供核对规则，不是待校验数据。"
        "下游只返回包含 valid 和 errors 字段的 JSON，不增加解释。",
        "## 指令层\n校验记录，必须遵守参考规范中的校验规则。\n\n"
        "## 情境层\n待校验数据：{records}\n\n"
        "## 参考层\n{validation_rule}：核对必须遵守的校验规则，不是待校验数据。\n\n"
        "## 输出层\n下游只返回包含 valid 和 errors 字段的 JSON，不增加解释。",
        "## 指令层\n校验记录。\n\n"
        "## 情境层\n待校验数据：{records}\n必须遵守参考规范中的校验规则。\n"
        "下游只返回包含 valid 和 errors 字段的 JSON，不增加解释。\n\n"
        "## 参考层\n{validation_rule}：核对必须遵守的校验规则，不是待校验数据。",
        ("遵守规范的要求在指令，规范来源在参考，记录在情境。", "下游 JSON 约定在输出，优化稿本身仍可分层。"),
        (("{records}", "情境层"), ("{validation_rule}", "参考层")),
    ),
    BoundaryCase(
        "lb-dual-purpose",
        "用附件 {handbook} 为新入职的档案员编写交接指南。事实只依据这份手册，同时借鉴其章节顺序；"
        "手册原文只保留一份，另处用交叉引用说明用途。交付中文 Markdown 指南，附依据位置。",
        "## 指令层\n编写交接指南，事实只依据情境中的手册；原文只保留一份，另处用交叉引用。\n\n"
        "## 情境层\n事实依据及唯一手册原文：{handbook}\n\n"
        "## 参考层\n借鉴情境中同一份手册的章节顺序。\n\n"
        "## 输出层\n面向新入职的档案员，交付中文 Markdown 指南，附依据位置。",
        "## 指令层\n编写交接指南，事实只依据下述手册；原文只保留一份，另处用交叉引用。\n\n"
        "## 参考层\n事实依据及唯一手册原文：{handbook}；同时借鉴其章节顺序。\n\n"
        "## 输出层\n面向新入职的档案员，交付中文 Markdown 指南，附依据位置。",
        ("唯一手册作为事实依据在情境，参考层交叉引用其章节顺序。", "保留档案员受众、中文指南及依据位置。"),
        (("{handbook}", "情境层"),),
    ),
    BoundaryCase(
        "lb-long-context",
        "根据下面的采样记录提出核查顺序；所有记录均待核查，不能当成已确认故障。"
        "只分析，不改任何设备配置。先区分异常线索与假设，再给核查建议。"
        "{table_layout} 仅参考表格版式。最终用中文表格列出线索、依据和建议。\n采样记录：\n" + _MEASUREMENTS,
        "## 指令层\n提出核查顺序，只分析，不改任何设备配置；不把待核查记录当成已确认故障。\n"
        "1. 区分异常线索与假设。\n2. 给出核查建议。\n\n"
        "## 情境层\n采样记录均待核查：\n" + _MEASUREMENTS + "\n\n"
        "## 参考层\n{table_layout}：仅参考表格版式。\n\n"
        "## 输出层\n用中文表格列出线索、依据和建议。",
        "## 指令层\n提出核查顺序，只分析，不改任何设备配置；不把待核查记录当成已确认故障。\n"
        "1. 区分异常线索与假设。\n2. 给出核查建议。\n\n"
        "## 情境层\n{table_layout}：仅参考表格版式。\n\n"
        "## 参考层\n采样记录均待核查：\n" + _MEASUREMENTS + "\n\n"
        "## 输出层\n用中文表格列出线索、依据和建议。",
        ("采样记录在情境，表格版式在参考，待核查状态保留。", "保留只分析、不改配置及先区分再建议的顺序。"),
        (("{table_layout}", "参考层"), ("记录 01", "情境层")),
    ),
    BoundaryCase(
        "lb-simple",
        "把 'Rain stopped before noon.' 翻译成中文，只给译文。",
        "将 'Rain stopped before noon.' 翻译为中文，只输出译文。",
        None,
        ("简单翻译保持简短，不强加四层标题、步骤或可选材料。", "保留原句、中文与只给译文的要求。"),
    ),
)
