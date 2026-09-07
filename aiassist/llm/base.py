"""LLM 客户端抽象：引擎层只依赖此接口，不依赖任何具体模型实现。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    """模型输出的一次结构化工具调用。"""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class LLMResponse:
    """模型返回结果。"""
    content: str = ""                 # 最终答案时非空
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"       # stop | tool_calls | length


class LLMClient(ABC):
    """模型交互抽象。"""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """多轮推理：传入上下文消息与工具 schema，返回文本或工具调用。

        messages 为 OpenAI 风格：{"role", "content", 可选 "tool_calls"/"tool_call_id"}。
        tools 为 JSON Schema 列表。
        on_delta：流式输出回调，收到文本增量时逐段调用（不支持流式的实现可一次性回调）。
        """
        raise NotImplementedError
