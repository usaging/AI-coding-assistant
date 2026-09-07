"""可观测性：执行轨迹结构化记录 + JSON 导出。

记录每次运行的：查询、模型、终止原因、每步工具调用与耗时、token 消耗、估算成本。
用于调试（"为什么这一步这么做"）与评估（成本/耗时统计）。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..tokens import estimate_cost
from .types import AgentResult


@dataclass
class RunTrace:
    query: str
    model: str
    final_answer: str
    terminated_by: str
    steps: list[dict]
    total_tokens_in: int
    total_tokens_out: int
    cost: float
    duration: float

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p


def build_trace(result: AgentResult, query: str, model: str, started_at: float) -> RunTrace:
    steps = [
        {
            "step_no": s.step_no,
            "duration": round(s.duration, 4),
            "tokens_in": s.tokens_in,
            "tokens_out": s.tokens_out,
            "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in s.tool_calls],
            "results": [
                {"success": r.success, "summary": (r.content or r.error or "")[:200]}
                for r in s.results
            ],
        }
        for s in result.steps
    ]
    cost = estimate_cost(model, result.total_tokens_in, result.total_tokens_out)
    return RunTrace(
        query=query,
        model=model,
        final_answer=result.final_answer,
        terminated_by=result.terminated_by,
        steps=steps,
        total_tokens_in=result.total_tokens_in,
        total_tokens_out=result.total_tokens_out,
        cost=cost,
        duration=round(time.monotonic() - started_at, 4),
    )
