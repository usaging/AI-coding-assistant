"""工具注册表：引擎通过它获取工具，实现"调度逻辑与具体工具解耦"。"""
from __future__ import annotations

from .base import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self

    def register_many(self, tools: list[BaseTool]) -> "ToolRegistry":
        for t in tools:
            self.register(t)
        return self

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list(self) -> list[BaseTool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        """所有工具的 schema，注入 system prompt。"""
        return [t.to_schema() for t in self._tools.values()]
