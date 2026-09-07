"""引擎层核心：ReAct Query Loop。

循环 = 模型推理 → 工具选择 → 执行（只读并发/写入串行）→ 结果回填 → 终止判断。
终止条件：显式 finish / max_steps / 连续无进展 / 异常（工具异常已在执行器兜底）。

- 流式输出：通过 on_event 回调发出 text_delta / step_start / tool_result / final 事件。
- 可观测性：记录每步耗时与 token 消耗，累计进 AgentResult。
"""
from __future__ import annotations

import json
import time
from typing import Callable

from ..config import AgentConfig
from ..llm.base import LLMClient
from ..memory.manager import MemoryManager
from ..tokens import count_tokens
from ..tools.executor import ToolExecutor
from ..tools.registry import ToolRegistry
from .events import Event
from .prompts import build_system_prompt
from .types import AgentResult, AgentStep


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        memory: MemoryManager,
        config: AgentConfig,
        working_dir: str = ".",
    ):
        self.llm = llm
        self.registry = registry
        self.memory = memory
        self.config = config
        self.working_dir = working_dir
        self.executor = ToolExecutor(registry)
        self._system_seeded = False

    def _seed_system(self, recall_text: str = "") -> None:
        """只在首轮注入一次 system prompt（含工具 schema 与冷记忆召回）。"""
        if self._system_seeded:
            return
        prompt = build_system_prompt(self.working_dir, self.registry.schemas())
        if recall_text:
            prompt += "\n\n[跨会话记忆] 以下历史记忆可能相关，可作参考：\n" + recall_text
        self.memory.add_message("system", prompt)
        self._system_seeded = True

    async def run(
        self,
        user_query: str,
        recall_top_k: int = 3,
        on_event: Callable[[Event], None] | None = None,
    ) -> AgentResult:
        def emit(type_: str, data=None) -> None:
            if on_event:
                on_event(Event(type_, data))

        # 跨会话复用：冷记忆 grep 召回
        hits = self.memory.recall(user_query, recall_top_k)
        recall_text = "\n".join(f"- {e.content}" for e, _ in hits)
        self._seed_system(recall_text)

        self.memory.add_message("user", user_query)

        steps: list[AgentStep] = []
        terminated_by = "max_steps"
        last_sig: tuple | None = None
        no_progress = 0
        total_in = total_out = 0

        def _on_delta(text: str) -> None:
            emit("text_delta", text)

        for step_no in range(1, self.config.max_steps + 1):
            emit("step_start", step_no)
            started_at = time.monotonic()

            # 1) 上下文管理：超预算则压缩
            self.memory.evict_if_needed()

            # 2) 模型推理（流式）
            messages = self.memory.build_messages()
            tokens_in = sum(count_tokens(str(m.get("content", ""))) for m in messages)
            response = await self.llm.chat(messages, self.registry.schemas(), on_delta=_on_delta)
            tokens_out = count_tokens(response.content)
            total_in += tokens_in
            total_out += tokens_out

            # 3) 无工具调用 → 最终答案，终止
            if not response.tool_calls:
                terminated_by = "finish"
                final = response.content or "(空回答)"
                self.memory.add_message("assistant", final)
                emit("final", final)
                return AgentResult(final, steps, terminated_by, total_in, total_out)

            # 4) 工具调用：记录 assistant 消息（带 tool_calls）
            call_ids = [tc.id or f"call_{step_no}_{i}" for i, tc in enumerate(response.tool_calls)]
            self.memory.add_message(
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for cid, tc in zip(call_ids, response.tool_calls)
                ],
            )

            # 5) 工具执行（只读并发 / 写入串行）
            results = await self.executor.execute(response.tool_calls)

            # 6) 结果回填（observation）+ 事件
            for cid, call, res in zip(call_ids, response.tool_calls, results):
                self.memory.add_message(
                    "tool", res.to_observation(),
                    tool_call_id=cid, name=call.name,
                )
                emit("tool_result", {
                    "name": call.name,
                    "arguments": call.arguments,
                    "success": res.success,
                    "content": (res.content or "")[:200],
                    "error": res.error,
                })

            steps.append(AgentStep(
                step_no, response.tool_calls, results,
                started_at=started_at, finished_at=time.monotonic(),
                tokens_in=tokens_in, tokens_out=tokens_out,
            ))

            # 7) 连续无进展检测：同签名连续 N 轮判定卡死
            sig = self._progress_signature(response.tool_calls, results)
            if sig == last_sig:
                no_progress += 1
            else:
                no_progress = 0
                last_sig = sig
            if no_progress >= self.config.no_progress_limit:
                terminated_by = "no_progress"
                break

        # 步数耗尽/无进展：强制收尾
        final = await self._force_finalize(_on_delta)
        emit("final", final)
        return AgentResult(final, steps, terminated_by, total_in, total_out)

    async def _force_finalize(self, on_delta=None) -> str:
        self.memory.add_message("user", "[系统] 请基于已有信息直接给出最终结论，不要继续调用工具。")
        resp = await self.llm.chat(self.memory.build_messages(), tools=[], on_delta=on_delta)
        return resp.content or "(无结论)"

    @staticmethod
    def _progress_signature(calls, results) -> tuple:
        parts = []
        for c, r in zip(calls, results):
            parts.append(f"{c.name}:{sorted(c.arguments.items())}:{r.success}:{r.content[:40]}")
        return tuple(parts)
