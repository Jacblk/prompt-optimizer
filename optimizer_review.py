"""Official OpenEvals review, using the engine's existing request accounting."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from importlib.metadata import version
import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context
from pydantic import PrivateAttr

from optimizer_config import ConfigurationError, OptimizerError
from optimizer_models import ModelReply, OutputError, Review, parse_output


ReviewRequest = Callable[[str, dict], Awaitable[ModelReply]]


def _evaluator_factory():
    # Load the review backend only when preparing an independently reviewed run.
    try:
        from openevals.llm import create_async_llm_as_judge
    except ImportError:
        raise ConfigurationError("独立评审需要 OpenEvals；请安装 requirements.txt 中的依赖。") from None
    return create_async_llm_as_judge


def require_review_backend():
    """Fail before spending generation requests if the backend is unavailable."""
    _evaluator_factory()
    return {"name": "openevals", "version": version("openevals")}


class _BudgetedJudge(BaseChatModel):
    """A LangChain model that delegates transport, retries and usage to the engine.

    Structured output is validated locally so Chat Completions gateways need not
    implement JSON Schema or tool calling. There is no additional SDK client.
    """

    _request: ReviewRequest = PrivateAttr()

    def __init__(self, request: ReviewRequest):
        super().__init__(cache=False, callbacks=[])
        self._request = request

    @property
    def _llm_type(self):
        return "optimizer_budgeted_judge"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("评审适配器仅支持异步调用。")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if (len(messages) != 2 or not isinstance(messages[0], SystemMessage)
                or not isinstance(messages[1], HumanMessage)
                or not isinstance(messages[0].content, str)
                or not isinstance(messages[1].content, str)):
            raise OutputError("评审消息结构不符合适配要求。")
        try:
            payload = json.loads(messages[1].content)
        except (ValueError, TypeError, RecursionError):
            raise OutputError("评审输入不是有效 JSON。") from None
        if not isinstance(payload, dict):
            raise OutputError("评审输入必须是 JSON 对象。")
        reply = await self._request(messages[0].content, payload)
        usage = None
        if all(type(n) is int and n >= 0 for n in
               (reply.input_tokens, reply.output_tokens, reply.total_tokens)):
            usage = {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                     "total_tokens": reply.total_tokens}
        message = AIMessage(content=reply.text, usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        if schema is not Review or include_raw or kwargs:
            raise ConfigurationError("评审适配器只支持现有 Review 结构。")
        return self | RunnableLambda(lambda message: parse_output(message.content, Review).model_dump())


async def evaluate_review(system: str, payload: dict, request: ReviewRequest) -> Review:
    """Run the official evaluator once, preserving literal prompts and strict JSON."""
    create_evaluator = _evaluator_factory()

    def messages(*, inputs=None, outputs=None, reference_outputs=None, request_payload):
        # A callable prompt bypasses template interpolation of JSON and {variables}.
        return [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)}]

    try:
        # This scope also covers OpenEvals' own traceable wrappers, not only the SDK.
        with tracing_context(enabled=False):
            evaluator = create_evaluator(prompt=messages, judge=_BudgetedJudge(request),
                                         output_schema=Review, feedback_key="prompt_review")
            value = await evaluator(request_payload=payload)
        return parse_output(json.dumps(value, ensure_ascii=False), Review)
    except OptimizerError:
        raise
    except Exception:
        # Framework errors may contain material or provider details.
        raise OutputError("OpenEvals 评审未完成；框架或服务详情未写入日志。") from None
