"""验证资源级依赖 DAG 执行器：独立调用并行、同资源串行、参数校验。"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiassist.llm.base import ToolCall
from aiassist.tools.base import BaseTool, SideEffect, ToolResult
from aiassist.tools.executor import ToolExecutor
from aiassist.tools.registry import ToolRegistry


def _resource_tool(name, side, resource, delay, events):
    class T(BaseTool):
        side_effect = side
        reads_params = ["path"] if side == SideEffect.READ else []
        writes_params = ["path"] if side == SideEffect.WRITE else []

        @property
        def name(self):
            return name

        def __init__(self):
            self.delay = delay
            self.events = events

        async def run(self, path="", **kwargs):
            self.events.append((name, "start", time.monotonic()))
            await asyncio.sleep(self.delay)
            self.events.append((name, "end", time.monotonic()))
            return ToolResult.ok(name)

    return T()


def _span(events, name):
    s = [e for e in events if e[0] == name and e[1] == "start"][0][2]
    e = [e for e in events if e[0] == name and e[1] == "end"][0][2]
    return s, e


def test_unrelated_read_not_blocked_by_write():
    """旧启发式会强制 read_C 等 write_B；DAG 证明无冲突后 read_C 提前并行。"""
    events = []
    reg = ToolRegistry().register_many([
        _resource_tool("write_B", SideEffect.WRITE, "B", 0.10, events),
        _resource_tool("read_A", SideEffect.READ, "A", 0.10, events),
        _resource_tool("read_C", SideEffect.READ, "C", 0.10, events),
    ])
    ex = ToolExecutor(reg)
    calls = [
        ToolCall("write_B", {"path": "B"}),
        ToolCall("read_A", {"path": "A"}),
        ToolCall("read_C", {"path": "C"}),
    ]
    results = asyncio.run(ex.execute(calls))
    assert [r.content for r in results] == ["write_B", "read_A", "read_C"]  # 结果保序

    ra_s, ra_e = _span(events, "read_A")
    rc_s, rc_e = _span(events, "read_C")
    wb_s, wb_e = _span(events, "write_B")
    assert ra_s < rc_e and rc_s < ra_e, "read_A 与 read_C 应并行"
    assert rc_e <= wb_s, "无关的 read_C 不应被 write_B 阻塞（关键优化点）"


def test_same_resource_read_after_write_is_ordered():
    """读写的资源相同 → 必须串行：write 先于 read。"""
    events = []
    reg = ToolRegistry().register_many([
        _resource_tool("write_B", SideEffect.WRITE, "B", 0.05, events),
        _resource_tool("read_B", SideEffect.READ, "B", 0.05, events),
    ])
    ex = ToolExecutor(reg)
    results = asyncio.run(ex.execute([
        ToolCall("write_B", {"path": "B"}),
        ToolCall("read_B", {"path": "B"}),
    ]))
    assert [r.content for r in results] == ["write_B", "read_B"]
    _, wb_e = _span(events, "write_B")
    rb_s, _ = _span(events, "read_B")
    assert wb_e <= rb_s, "同资源的 read 必须等 write 完成"


def test_param_validation_fails_gracefully():
    reg = ToolRegistry().register_many([
        _resource_tool("read_A", SideEffect.READ, "A", 0.01, []),
    ])
    ex = ToolExecutor(reg)
    # read_A 声明 required=["path"] 之外，这里直接缺参 —— 用带 required 的工具验证
    class StrictRead(BaseTool):
        name = "strict_read"
        side_effect = SideEffect.READ
        reads_params = ["path"]
        parameters = {"type": "object",
                      "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                      "required": ["path"]}

        async def run(self, **kw):
            return ToolResult.ok("ok")

    reg2 = ToolRegistry().register(StrictRead())
    ex2 = ToolExecutor(reg2)
    results = asyncio.run(ex2.execute([
        ToolCall("strict_read", {}),                        # 缺 path
        ToolCall("strict_read", {"path": "a", "limit": "x"}),  # limit 类型错误
    ]))
    assert results[0].success is False and "必填" in results[0].error
    assert results[1].success is False and "limit" in results[1].error


if __name__ == "__main__":
    test_unrelated_read_not_blocked_by_write()
    test_same_resource_read_after_write_is_ordered()
    test_param_validation_fails_gracefully()
    print("test_executor_dag OK")
