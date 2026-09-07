"""OpenAI 客户端：封装 function calling + 流式输出，统一为 LLMResponse。"""
from __future__ import annotations

from typing import Any, Callable

from .base import LLMClient, LLMResponse, ToolCall


class OpenAIClient(LLMClient):
    def __init__(self, model: str, api_key: str, base_url: str | None = None, temperature: float = 0.2, extra_body: dict | None = None):
        self.model = model
        self.temperature = temperature
        self.extra_body = extra_body or {}
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # 依赖可选：mock 模式无需安装
            raise RuntimeError("使用 openai provider 需要 `pip install openai`") from e
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    async def chat(self, messages: list[dict], tools: list[dict], on_delta: Callable[[str], None] | None = None) -> LLMResponse:
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or None,
            temperature=self.temperature,
            stream=True,
            extra_body=self.extra_body or None,  # 透传 thinking/reasoning_effort 等模型扩展参数
        )
        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        finish_reason = "stop"
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta and delta.content:
                content_parts.append(delta.content)
                if on_delta:
                    on_delta(delta.content)
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    acc = tool_calls_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            acc["name"] = tc.function.name
                        if tc.function.arguments:
                            acc["arguments"] += tc.function.arguments

        tool_calls = [
            ToolCall(id=acc["id"], name=acc["name"], arguments=_safe_json(acc["arguments"]))
            for _, acc in sorted(tool_calls_acc.items())
        ]
        return LLMResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
        )


def _safe_json(s: str) -> dict:
    import json
    try:
        return json.loads(s or "{}")
    except json.JSONDecodeError:
        return {}
