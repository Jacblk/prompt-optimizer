"""Offline selection of built-in teaching examples; never selects user material."""
from __future__ import annotations

from copy import deepcopy
import json
import re

from langchain_core.example_selectors import BaseExampleSelector, LengthBasedExampleSelector
from langchain_core.prompts import PromptTemplate

from optimizer_config import ConfigurationError
from optimizer_examples import GENERATION_EXAMPLES


DEFAULT_MAX_BUILTIN_EXAMPLES = 3
DEFAULT_MAX_EXAMPLE_CHARS = 4000
SELECTOR_VERSION = "builtin-rule-length-v2"

# Profiles follow GENERATION_EXAMPLES in order. IDs are local report labels and
# are not included in the demonstrations sent to a model.
EXAMPLE_PROFILES = (
    ("simple_explanation", {"general"}),
    ("ordered_explanation", {"explanation", "ordered", "audience"}),
    ("label_mapping", {"classification", "template"}),
    ("adopted_json_reference", {"json", "input_output_reference"}),
    ("comparison", {"comparison"}),
    ("conflicting_reference", {"conflict"}),
    ("diagnosis", {"diagnosis", "readonly"}),
    ("calculation", {"calculation"}),
    ("code_reference", {"code", "implementation_reference"}),
    ("file_reference", {"artifact", "artifact_reference"}),
    ("natural_material_task", {"external_material"}),
    ("natural_material_goal_missing", {"external_material"}),
)

_SIGNALS = {
    "code": r"代码|函数|脚本|编程|补丁|编译|单元测试|\b(?:source_code|code|function|script|patch|implement|refactor|debug|python|javascript|typescript|sql)\b",
    "classification": r"标签|分类|打标|\b(?:classify|classification|label)\b",
    "comparison": r"比较|对比|权衡|推荐|选一个|\b(?:compare|comparison|recommend|choose)\b",
    "conflict": r"冲突|矛盾|不一致|\b(?:conflict|contradict\w*)\b",
    "diagnosis": r"排查|诊断|排障|定位|日志|报错|故障|变慢|\b(?:troubleshoot\w*|diagnos\w*|logs?|traceback)\b",
    "calculation": r"计算|核算|总额|单价|折扣|推导|逻辑|命题|\b(?:calculat\w*|arithmetic|deduc\w*|logic)\b",
    "explanation": r"解释|讲解|概念|\b(?:explain|explanation)\b",
    "ordered": r"步骤|顺序|逐步|先.+再|\b(?:steps?|step-by-step)\b",
    "audience": r"新手|初学|刚上岗|实习|\b(?:beginner|novice)\b",
    "readonly": r"只读|仅检查|只检查|不改|不修改|不运行|\b(?:read-only|inspect only)\b",
    "json": r"\bjson\b",
    "template": r"模板|\{[\w]+\}|\btemplate\b",
    # Teaching hint only; semantic role/presence is still recognized by the
    # layer model. A mention here never loads a file or supplies task facts.
    "external_material": (
        r"(?:所给|所附|给定|提供|上传|上述|上面|下面|以下|这|那)(?:的)?"
        r"(?:[一二三四五六七八九十两几\d]+[份张段个条])?"
        r"(?:视频|图片|图像|截图|音频|录音|附件|文件|材料|资料|网页|网址|链接|pdf)"
        r"|上图|下图|附件|https?://"
        r"|\b(?:attached|uploaded|provided|above|following)\s+"
        r"(?:videos?|images?|pictures?|screenshots?|audio|recordings?|attachments?|files?|documents?|links?|urls?)\b"
    ),
}
_WEIGHTS = {feature: 8 for feature in (
    "code", "artifact", "classification", "comparison", "diagnosis", "calculation",
    "input_output_reference", "implementation_reference", "artifact_reference", "external_material",
)} | {"conflict": 12, "explanation": 4, "readonly": 3, "json": 3,
     "template": 2, "ordered": 2, "audience": 2}
_PRIMARY_FEATURES = set(_WEIGHTS) - {"template", "ordered", "audience"}


def validate_example_limits(max_examples, max_chars):
    if (type(max_examples) is not int or max_examples < 0
            or type(max_chars) is not int or max_chars < 0):
        raise ConfigurationError("内置示范数量和字符上限须为非负整数；0 表示不注入示范。")


def _features(inputs):
    # Code inside a quoted block is material, not sufficient evidence that the
    # current task is programming. This is a deterministic heuristic, not an
    # intent classifier or a permission check.
    request = re.sub(r"```[\s\S]*?```", "", inputs.get("original_request", "")).lower()
    simple_transform = re.match(r"\s*(?:请|帮我)?\s*(?:翻译|润色|translate\b|proofread\b)", request)
    features = set() if simple_transform else {
        name for name, pattern in _SIGNALS.items() if re.search(pattern, request)
    }
    if not simple_transform:
        artifact_text = re.sub(r"(?:不|不要|不得|禁止|无需)(?:生成|创建|保存|写入|导出)[^，。；\n]*", "", request)
        artifact_text = re.sub(r"\b(?:do not|don't|never)\s+(?:create|generate|save|write|export)\b[^.;\n]*", "", artifact_text)
        if (re.search(r"报告|文档|文件|\b(?:markdown|report|document|file|pdf|csv|xlsx|pptx|docx)\b", artifact_text)
                and re.search(r"生成|创建|制作|导出|保存|写入|交付|\b(?:create|generate|save|write|export|deliver)\b", artifact_text)):
            features.add("artifact")
    if inputs.get("adopted_reference") == "yes":
        kinds = set(inputs.get("reference_kinds", "").split(","))
        if "implementation" in kinds:
            features.add("implementation_reference")
        if "artifact" in kinds:
            features.add("artifact_reference")
        if "execution" in kinds:
            features.add("diagnosis")
        if "input_output" in kinds or (not kinds - {""} and not features & {"code", "artifact"}):
            features.add("input_output_reference")
    return features


class BuiltinExampleSelector(BaseExampleSelector):
    """Rank a fixed teaching library, then select whole examples by character length."""

    def __init__(self, *, max_examples=DEFAULT_MAX_BUILTIN_EXAMPLES,
                 max_chars=DEFAULT_MAX_EXAMPLE_CHARS):
        validate_example_limits(max_examples, max_chars)
        self.max_examples, self.max_chars = max_examples, max_chars
        self._entries = [
            {"example_id": example_id, "tags": tags, "example": example,
             "serialized": json.dumps(example, ensure_ascii=False)}
            for (example_id, tags), example in zip(EXAMPLE_PROFILES, GENERATION_EXAMPLES, strict=True)
        ]

    def add_example(self, example):
        raise TypeError("内置示范库固定；用户参考材料通过原始需求或补层选择传递。")

    def select_with_metadata(self, input_variables):
        features = _features(input_variables)
        ranked = sorted(
            self._entries,
            key=lambda entry: -sum(_WEIGHTS.get(tag, 0) for tag in entry["tags"] & features),
        )
        # A variable/step/audience hint can rank an already relevant example,
        # but cannot turn a programming task into a classification task.
        relevant = [entry for entry in ranked if entry["tags"] & features & _PRIMARY_FEATURES]
        # One short example covers unknown/simple tasks and provides a fallback
        # when a relevant example cannot fit. We do not fill spare space with
        # unrelated examples merely to reach the count limit.
        candidates = relevant + [self._entries[0]]
        candidates = [entry for entry in candidates if len(entry["serialized"]) + 2 <= self.max_chars]
        if not self.max_examples or not self.max_chars:
            selected = []
        else:
            limiter = LengthBasedExampleSelector(
                examples=candidates,
                example_prompt=PromptTemplate.from_template("{serialized}, "),
                # JSON array brackets and ', ' separators cost exactly two
                # characters per item. No word-count approximation for Chinese.
                get_text_length=len,
                max_length=self.max_chars,
            )
            # This budget applies only to the teaching block, not to user input.
            selected = limiter.select_examples({})[:self.max_examples]
        examples = [deepcopy(entry["example"]) for entry in selected]
        metadata = {
            "selector": SELECTOR_VERSION,
            "example_ids": [entry["example_id"] for entry in selected],
            "example_count": len(examples),
            "example_characters": len(json.dumps(examples, ensure_ascii=False)) if examples else 0,
            "available_example_count": len(self._entries),
            "all_example_characters": len(json.dumps(GENERATION_EXAMPLES, ensure_ascii=False)),
            "max_builtin_examples": self.max_examples, "max_example_chars": self.max_chars,
            "matched_features": sorted(features),
        }
        return examples, metadata

    def select_examples(self, input_variables):
        return self.select_with_metadata(input_variables)[0]
