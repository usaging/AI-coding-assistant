"""Mock LLM 客户端：按脚本队列返回预设响应，并模拟流式输出。

用途：
1. 离线 demo —— 不需要 API key 也能完整演示 ReAct 循环 + 工具执行 + 结果回填 + 流式。
2. 单元测试 —— 确定性输出，便于断言循环行为和终止条件。
"""
from __future__ import annotations

from .base import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    def __init__(self, responses: list[LLMResponse] | None = None, chunk_size: int = 12):
        self._responses = list(responses or [])
        self._idx = 0
        self.chunk_size = chunk_size
        self.calls: list[dict] = []  # 记录每次实际收到的上下文，便于断言

    async def chat(self, messages: list[dict], tools: list[dict], on_delta=None) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
        else:
            # 脚本耗尽：返回终止答案，防止死循环
            resp = LLMResponse(content="(mock) 已达到预设脚本末尾，终止执行。", finish_reason="stop")

        # 模拟流式：按 chunk 逐段回调
        if on_delta and resp.content:
            for i in range(0, len(resp.content), self.chunk_size):
                on_delta(resp.content[i:i + self.chunk_size])
        return resp
