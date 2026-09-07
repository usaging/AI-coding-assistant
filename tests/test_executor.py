"""验证工具执行器：只读并发、写入串行、顺序保序。"""
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


def _make_tool(name: str, side: SideEffect, delay: float, events: list):
    class T(BaseTool):
        def __init__(self):
            self.name_ = name
            self.side_effect = side
            self.delay = delay
            self.events = events

        @property
        def name(self):  # 简化：用属性覆盖类属性
            return self.name_

        async def run(self, **kwargs) -> ToolResult:
            self.events.append((self.name_, "start", time.monotonic()))
            await asyncio.sleep(self.delay)
            self.events.append((self.name_, "end", time.monotonic()))
            return ToolResult.ok(self.name_)

    return T()


def test_read_parallel_write_serial_order():
    events: list[tuple] = []
    reg = ToolRegistry()
    reg.register_many([
        _make_tool("read_a", SideEffect.READ, 0.10, events),
        _make_tool("read_b", SideEffect.READ, 0.10, events),
        _make_tool("write_a", SideEffect.WRITE, 0.05, events),
        _make_tool("read_c", SideEffect.READ, 0.05, events),
        _make_tool("write_b", SideEffect.WRITE, 0.05, events),
    ])
    ex = ToolExecutor(reg)
    calls = [ToolCall(n) for n in ("read_a", "read_b", "write_a", "read_c", "write_b")]

    results = asyncio.run(ex.execute(calls))

    # 1) 结果顺序与调用顺序一致
    assert [r.content for r in results] == ["read_a", "read_b", "write_a", "read_c", "write_b"]

    # 2) read_a / read_b 并行：二者时间区间重叠
    def span(name):
        s = [e for e in events if e[0] == name and e[1] == "start"][0][2]
        e = [e for e in events if e[0] == name and e[1] == "end"][0][2]
        return s, e

    ra_s, ra_e = span("read_a")
    rb_s, rb_e = span("read_b")
    assert ra_s < rb_e and rb_s < ra_e, "read_a 与 read_b 应并行执行"

    # 3) write 串行保序：write_a 结束后 read_c 才开始；read_c 结束后 write_b 才开始
    wa_s, wa_e = span("write_a")
    rc_s, rc_e = span("read_c")
    wb_s, wb_e = span("write_b")
    assert wa_e <= rc_s, "write_a 应在 read_c 之前完成（write 后的 read 能看到写入结果）"
    assert rc_e <= wb_s, "read_c 应在 write_b 之前完成"


def test_unknown_tool_returns_error_not_raise():
    ex = ToolExecutor(ToolRegistry())
    results = asyncio.run(ex.execute([ToolCall("not_exist")]))
    assert len(results) == 1
    assert results[0].success is False
    assert "未注册" in results[0].error


if __name__ == "__main__":
    test_read_parallel_write_serial_order()
    test_unknown_tool_returns_error_not_raise()
    print("test_executor OK")
