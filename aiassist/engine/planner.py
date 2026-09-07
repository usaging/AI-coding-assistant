"""Plan-and-Execute 规划器。

解决纯 ReAct 的短板：复杂任务多轮盲目试探会丢失全局观、浪费 token。
规划器先把任务拆成有序子任务，再对每个子任务复用 ReAct Agent 独立执行，最后汇总。

- Planner.make_plan(query)：调用 LLM 产出 JSON 子任务数组。
- PlanAndExecuteAgent：规划 → 逐项执行（每个子任务一个独立 Agent）→ 汇总结果。
"""
from __future__ import annotations

import json
import re
from typing import Callable

from ..llm.base import LLMClient
from .agent import Agent
from .events import Event

PLAN_PROMPT = """你是任务规划器。把下面的用户需求拆解为 2-6 个可独立执行的子任务，按依赖顺序排列。
每个子任务一句话、可操作（例如"定位配置文件""阅读某函数实现""总结并给出结论"）。
只输出 JSON 数组，不要输出任何其它内容。

示例输出：["定位相关文件", "阅读核心实现", "总结架构并给出结论"]

用户需求：{query}
"""


class Planner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def make_plan(self, query: str, on_event: Callable[[Event], None] | None = None) -> list[str]:
        resp = await self.llm.chat(
            [{"role": "user", "content": PLAN_PROMPT.format(query=query)}],
            tools=[],
        )
        steps = self._parse(resp.content)
        if on_event:
            on_event(Event("plan", steps))
        return steps

    @staticmethod
    def _parse(content: str) -> list[str]:
        m = re.search(r"\[.*\]", content, re.S)
        if not m:
            return [content.strip()] if content.strip() else []
        try:
            data = json.loads(m.group(0))
            return [str(x) for x in data if str(x).strip()]
        except json.JSONDecodeError:
            return [content.strip()] if content.strip() else []


class PlanAndExecuteAgent:
    def __init__(self, planner: Planner, agent_factory: Callable[[], Agent]):
        self.planner = planner
        self.agent_factory = agent_factory

    async def run(self, query: str, on_event: Callable[[Event], None] | None = None) -> dict:
        def emit(type_: str, data=None) -> None:
            if on_event:
                on_event(Event(type_, data))

        plan = await self.planner.make_plan(query, on_event=on_event)
        if not plan:
            plan = [query]  # 规划失败则退化为直接执行

        results: list[dict] = []
        for i, task in enumerate(plan, 1):
            emit("subtask", {"index": i, "total": len(plan), "title": task})
            agent = self.agent_factory()          # 每个子任务独立 Agent（独立记忆/上下文）
            r = await agent.run(task, on_event=on_event)
            results.append({
                "task": task,
                "answer": r.final_answer,
                "steps": len(r.steps),
                "tokens_in": r.total_tokens_in,
                "tokens_out": r.total_tokens_out,
            })

        return {"plan": plan, "results": results}
