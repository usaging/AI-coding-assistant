"""评估 harness：用 MockLLMClient 驱动确定性任务，统计成功率 / token 成本 / 耗时。

评估对象是"Agent 循环是否按预期编排工具、是否正确终止、是否给出结论"，
脚本化模型行为保证可复现，不依赖真实模型或网络。
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiassist.bootstrap import build_agent
from aiassist.config import Config
from aiassist.llm import LLMResponse, MockLLMClient, ToolCall
from aiassist.tokens import estimate_cost


@dataclass
class EvalTask:
    id: str
    query: str
    script: list[LLMResponse]
    expect_tools: list[str]        # 期望依次调用的工具名
    answer_contains: str = ""


TASKS = [
    EvalTask(
        id="locate_and_read",
        query="定位并阅读核心类",
        script=[
            LLMResponse(tool_calls=[ToolCall(name="grep", arguments={"pattern": "class Agent"})]),
            LLMResponse(tool_calls=[ToolCall(name="read_file", arguments={"path": "aiassist/engine/agent.py", "limit": 40})]),
            LLMResponse(content="已定位 Agent 类并阅读其 ReAct 循环实现。"),
        ],
        expect_tools=["grep", "read_file"],
        answer_contains="Agent",
    ),
    EvalTask(
        id="single_read",
        query="读取 agent.py",
        script=[
            LLMResponse(tool_calls=[ToolCall(name="read_file", arguments={"path": "aiassist/engine/agent.py"})]),
            LLMResponse(content="文件内容已读取。"),
        ],
        expect_tools=["read_file"],
        answer_contains="读取",
    ),
    EvalTask(
        id="direct_answer",
        query="1+1 等于几",
        script=[LLMResponse(content="等于 2。")],
        expect_tools=[],
        answer_contains="2",
    ),
    EvalTask(
        id="unknown_tool_recovery",
        query="调用不存在的工具后改用 read_file",
        script=[
            LLMResponse(tool_calls=[ToolCall(name="not_exist", arguments={})]),
            LLMResponse(tool_calls=[ToolCall(name="read_file", arguments={"path": "aiassist/engine/agent.py"})]),
            LLMResponse(content="工具失败后已改用它法。"),
        ],
        expect_tools=["not_exist", "read_file"],
        answer_contains="改用它法",
    ),
]


async def run_task(task: EvalTask, working_dir: Path, model: str) -> dict:
    cfg = Config()
    llm = MockLLMClient(task.script)
    agent = build_agent(cfg, working_dir=str(working_dir), llm=llm)
    started = time.monotonic()
    result = await agent.run(task.query)

    got_tools = [c.name for s in result.steps for c in s.tool_calls]
    ok_tools = got_tools == task.expect_tools
    ok_answer = task.answer_contains in result.final_answer
    ok = ok_tools and ok_answer and result.terminated_by == "finish"

    return {
        "id": task.id,
        "ok": ok,
        "ok_tools": ok_tools,
        "ok_answer": ok_answer,
        "terminated_by": result.terminated_by,
        "tools": got_tools,
        "steps": len(result.steps),
        "tokens_in": result.total_tokens_in,
        "tokens_out": result.total_tokens_out,
        "duration": round(time.monotonic() - started, 4),
    }


async def main() -> int:
    working_dir = Path(__file__).resolve().parents[1]
    model = "gpt-4o-mini"
    results = []
    for t in TASKS:
        r = await run_task(t, working_dir, model)
        results.append(r)
        mark = "PASS" if r["ok"] else "FAIL"
        detail = "" if r["ok"] else f" (tools={r['tools']} expect={t.expect_tools}, term={r['terminated_by']})"
        print(f"[{mark}] {r['id']}: tools={r['tools']} steps={r['steps']}{detail}")

    passed = sum(1 for r in results if r["ok"])
    total_in = sum(r["tokens_in"] for r in results)
    total_out = sum(r["tokens_out"] for r in results)
    cost = estimate_cost(model, total_in, total_out)
    print(f"\n成功率: {passed}/{len(results)}")
    print(f"总 token 输入/输出: {total_in}/{total_out} | 估算成本: ${cost:.5f}")

    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存 {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
