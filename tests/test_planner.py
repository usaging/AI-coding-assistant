"""验证规划器（Plan-and-Execute）与可观测性轨迹。"""
from __future__ import annotations

import asyncio
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiassist.bootstrap import build_agent
from aiassist.config import Config
from aiassist.engine import PlanAndExecuteAgent, Planner, build_trace
from aiassist.engine.types import AgentResult
from aiassist.llm import LLMResponse, MockLLMClient

ROOT = Path(__file__).resolve().parents[1]


def test_planner_parse():
    p = Planner(None)
    assert p._parse('["a", "b", "c"]') == ["a", "b", "c"]
    assert p._parse('说明文字 ["x", "y"] 尾巴') == ["x", "y"]
    assert p._parse('无数组输出') == ["无数组输出"]
    assert p._parse('') == []


def test_plan_and_execute():
    planner_llm = MockLLMClient([LLMResponse(content='["读文件A", "写文件B"]')])
    planner = Planner(planner_llm)
    cfg = Config()

    def factory():
        # 每个子任务独立 Agent：mock 直接给结论
        return build_agent(cfg, working_dir=".", llm=MockLLMClient([LLMResponse(content="完成")]))

    pa = PlanAndExecuteAgent(planner, factory)
    result = asyncio.run(pa.run("测试任务"))

    assert result["plan"] == ["读文件A", "写文件B"]
    assert len(result["results"]) == 2
    assert all(r["answer"] == "完成" for r in result["results"])


def test_build_trace_roundtrip():
    result = AgentResult(
        final_answer="hi", steps=[], terminated_by="finish",
        total_tokens_in=100, total_tokens_out=50,
    )
    tr = build_trace(result, "query", "gpt-4o-mini", time.monotonic() - 1.0)
    d = tr.to_dict()
    assert d["total_tokens_in"] == 100
    assert d["total_tokens_out"] == 50
    assert d["cost"] > 0
    assert d["duration"] >= 0

    # 导出 JSON 落盘（workspace 内临时目录）
    tmp = ROOT / ("tmp-test-" + uuid.uuid4().hex[:8])
    tmp.mkdir()
    try:
        path = tr.save(tmp / "trace.json")
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["query"] == "query"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_planner_parse()
    test_plan_and_execute()
    test_build_trace_roundtrip()
    print("test_planner OK")
