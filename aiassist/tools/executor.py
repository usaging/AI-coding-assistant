"""工具执行器：资源级依赖 DAG + 副作用并发。

相比"顺序位置启发式"的升级点：
1. 每个工具声明资源读写集（reads_params/writes_params 对应的参数值）。
2. 对同一批 tool_calls 构建依赖 DAG：若 call i 写入了 call j 读取/写入的资源，则 i 先于 j。
3. 拓扑分层：层内无依赖 → 只读并行、写入串行；层间顺序执行。

效果：旧启发式下"write(B) 之后的 read(C)"即使读的是无关文件也强制等待；
现在依赖分析证明二者无资源冲突后，read(C) 可提前与 read(A) 并行，缩短任务耗时。

安全兜底：未声明资源参数的工具按"未知资源"（ANY）处理，与任何调用都产生依赖 → 保守串行。
"""
from __future__ import annotations

import asyncio
from collections import deque

from ..llm.base import ToolCall
from .base import BaseTool, SideEffect, ToolResult
from .registry import ToolRegistry
from .validator import validate_arguments

ANY = object()  # 未知资源哨兵


def _overlap(a: set, b: set) -> bool:
    return (ANY in a) or (ANY in b) or bool(a & b)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(self, calls: list[ToolCall]) -> list[ToolResult]:
        n = len(calls)
        results: list[ToolResult | None] = [None] * n
        tools: list[BaseTool | None] = [self.registry.get(c.name) for c in calls]

        # 1) 参数校验：缺参/错参 → 直接作为失败结果，不阻断其余调用
        for i, (call, tool) in enumerate(zip(calls, tools)):
            if tool is None:
                results[i] = ToolResult.fail(f"未注册的工具: {call.name}")
                continue
            err = validate_arguments(tool.parameters, call.arguments)
            if err:
                results[i] = ToolResult.fail(f"参数校验失败: {err}")

        # 2) 提取资源读写集（未声明的按未知资源保守处理）
        res: list[tuple[set, set]] = []
        for i, (call, tool) in enumerate(zip(calls, tools)):
            if tool is None or results[i] is not None:
                res.append((set(), set()))
                continue
            reads, writes = tool.resources(call.arguments)
            if tool.side_effect == SideEffect.READ and (not tool.reads_params or not reads):
                reads = {ANY}  # 未声明资源，或声明了但实参缺失（走默认整目录）→ 未知
            if tool.side_effect == SideEffect.WRITE and (not tool.writes_params or not writes):
                writes = {ANY}
            res.append((reads, writes))

        # 3) 依赖边：仅对原始顺序靠后的调用（j > i）建边。
        #    若 call i 写入了 call j 读取/写入的资源，则 j 依赖 i（i 先于 j）。
        #    "先读后写"（读在写之前）不产生依赖——读的是写入前的旧值，语义合法。
        edges: list[set[int]] = [set() for _ in range(n)]
        indeg = [0] * n
        for i in range(n):
            if results[i] is not None:
                continue
            wi = res[i][1]
            if not wi:
                continue  # i 无写入，不会成为任何调用的依赖前置
            for j in range(i + 1, n):
                if results[j] is not None:
                    continue
                rj, wj = res[j]
                if _overlap(wi, rj) or _overlap(wi, wj):
                    if j not in edges[i]:
                        edges[i].add(j)
                        indeg[j] += 1

        # 4) 拓扑分层（Kahn，层内节点互不依赖）
        levels = self._topo_levels(edges, indeg)

        # 5) 逐层执行：层内只读并行、写入串行
        for level in levels:
            read_idx = [i for i in level if results[i] is None and tools[i].side_effect == SideEffect.READ]
            write_idx = [i for i in level if results[i] is None and tools[i].side_effect == SideEffect.WRITE]

            read_results = await asyncio.gather(*(self._run(calls[i], tools[i]) for i in read_idx))
            write_results = await self._run_writes_serial([(i, calls[i], tools[i]) for i in write_idx])

            for i, r in zip(read_idx, read_results):
                results[i] = r
            for (i, _, _), r in zip([(i, calls[i], tools[i]) for i in write_idx], write_results):
                results[i] = r

        return [r for r in results if r is not None]

    async def _run(self, call: ToolCall, tool: BaseTool) -> ToolResult:
        try:
            return await tool.run(**call.arguments)
        except Exception as e:  # 工具内部异常兜底，不中断整个循环
            return ToolResult.fail(f"{type(e).__name__}: {e}")

    async def _run_writes_serial(self, items) -> list[ToolResult]:
        return [await self._run(call, tool) for (_, call, tool) in items]

    @staticmethod
    def _topo_levels(edges: list[set[int]], indeg: list[int]) -> list[list[int]]:
        levels: list[list[int]] = []
        queue = deque(i for i, d in enumerate(indeg) if d == 0)
        while queue:
            cur: list[int] = []
            for _ in range(len(queue)):
                node = queue.popleft()
                cur.append(node)
                for nb in edges[node]:
                    indeg[nb] -= 1
                    if indeg[nb] == 0:
                        queue.append(nb)
            levels.append(cur)
        return levels
