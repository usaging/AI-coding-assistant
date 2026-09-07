"""端到端验证：用 MockLLMClient 跑通完整 ReAct 循环。"""
from __future__ import annotations

import asyncio
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiassist.bootstrap import build_agent
from aiassist.config import Config
from aiassist.llm import LLMResponse, MockLLMClient, ToolCall

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def ws_tmpdir():
    d = ROOT / ("tmp-test-" + uuid.uuid4().hex[:8])
    d.mkdir(parents=True, exist_ok=False)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_full_loop_with_mock():
    # 脚本：grep 定位 → read_file 读内容 → 最终答案
    script = [
        LLMResponse(tool_calls=[ToolCall(name="grep", arguments={"pattern": "class Agent"})]),
        LLMResponse(tool_calls=[ToolCall(name="read_file", arguments={"path": "aiassist/engine/agent.py", "limit": 40})]),
        LLMResponse(content="已定位 Agent 类，实现了 ReAct Query Loop。"),
    ]
    with ws_tmpdir() as tmp:
        cfg = Config()
        cfg.memory.cold_dir = str(tmp / "memory")
        llm = MockLLMClient(script)
        agent = build_agent(cfg, working_dir=str(ROOT), llm=llm)

        result = asyncio.run(agent.run("介绍一下 Agent 的实现"))

        assert result.terminated_by == "finish"
        assert "Agent" in result.final_answer
        assert len(result.steps) == 2                      # 两轮工具调用
        assert result.steps[0].tool_calls[0].name == "grep"
        assert result.steps[0].results[0].success is True  # grep 命中
        assert result.steps[1].tool_calls[0].name == "read_file"
        assert result.steps[1].results[0].success is True  # 读到文件


def test_max_steps_terminates():
    # 脚本永远调用工具 → 触发 max_steps 终止
    script = [LLMResponse(tool_calls=[ToolCall(name="grep", arguments={"pattern": "x"})]) for _ in range(20)]
    cfg = Config()
    cfg.agent.max_steps = 3
    with ws_tmpdir() as tmp:
        cfg.memory.cold_dir = str(tmp / "memory")
        llm = MockLLMClient(script)
        agent = build_agent(cfg, working_dir=str(ROOT), llm=llm)
        result = asyncio.run(agent.run("循环任务"))
        assert result.terminated_by in ("max_steps", "no_progress")
        assert len(result.steps) <= cfg.agent.max_steps


if __name__ == "__main__":
    test_full_loop_with_mock()
    test_max_steps_terminates()
    print("test_agent_mock OK")
