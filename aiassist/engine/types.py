"""引擎层数据类型：执行轨迹与结果（含计时与 token 统计，用于可观测性）。"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import ToolCall
from ..tools.base import ToolResult


@dataclass
class Message:
    """通用消息（MVP 中热记忆直接承载，保留类型便于扩展）。"""
    role: str
    content: str
    tool_calls: list = field(default_factory=list)


@dataclass
class AgentStep:
    """一轮执行轨迹：可观测、可回放。"""
    step_no: int
    tool_calls: list[ToolCall]
    results: list[ToolResult]
    started_at: float = 0.0
    finished_at: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def duration(self) -> float:
        return self.finished_at - self.started_at


@dataclass
class AgentResult:
    final_answer: str
    steps: list[AgentStep]
    terminated_by: str = "finish"  # finish | max_steps | no_progress | error
    total_tokens_in: int = 0
    total_tokens_out: int = 0
