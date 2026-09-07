"""工具层：统一工具接口。

规范三要素：
1. 参数描述 —— 每个工具带 JSON Schema，注入 system prompt 供模型正确调用。
2. 执行结果 —— 统一返回 ToolResult，引擎回填 observation 格式一致。
3. 异常处理 —— 工具内部捕获异常转为 ToolResult(error=...)，而非抛出中断循环。

资源声明（依赖分析用）：
- reads_params / writes_params：声明哪些参数的值是"读取/写入"的资源（文件或目录路径）。
  执行器据此对同一批 tool_calls 做依赖 DAG 分析，独立调用并行、有依赖的串行。
  未声明的工具按"未知资源"保守串行。
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SideEffect(str, Enum):
    READ = "read"    # 只读、无副作用
    WRITE = "write"  # 有副作用，需串行且保序


@dataclass
class ToolResult:
    success: bool
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def ok(cls, content: str, **metadata) -> "ToolResult":
        return cls(success=True, content=content, metadata=metadata)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        return cls(success=False, error=error)

    def to_observation(self) -> str:
        """转成回填给模型的 observation 文本，格式统一。"""
        if self.success:
            return self.content or "(ok)"
        return f"[工具执行失败] {self.error}"


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    side_effect: SideEffect = SideEffect.READ

    # 资源读写声明：参数名列表，其值是被读/写的资源（路径）
    reads_params: list[str] = []
    writes_params: list[str] = []

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    def resources(self, arguments: dict[str, Any]) -> tuple[set[str], set[str]]:
        """从实参提取资源读写集（路径做归一化）。"""
        reads = {_norm(arguments[p]) for p in self.reads_params if p in arguments}
        writes = {_norm(arguments[p]) for p in self.writes_params if p in arguments}
        return reads, writes

    def to_schema(self) -> dict:
        """OpenAI function-calling 风格 schema，注入 system prompt。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _norm(v: Any) -> str:
    s = str(v)
    return os.path.normpath(s) if s else s
