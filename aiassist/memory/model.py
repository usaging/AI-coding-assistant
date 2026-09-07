"""记忆数据模型：热/温/冷三层 + 优先级评分。"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class MemoryTier(str, Enum):
    HOT = "hot"    # 当前会话上下文，直接拼进 prompt
    WARM = "warm"  # 会话级摘要，会话内快速检索
    COLD = "cold"  # 跨会话结构化事实，Markdown + grep 检索


@dataclass
class MemoryEntry:
    id: str
    kind: str                      # system/user/assistant/tool/summary/fact
    content: str
    tier: MemoryTier = MemoryTier.HOT
    priority: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    compression_level: int = 0     # 0=L0 原始 … 4=L4 结构化
    extra: dict = field(default_factory=dict)  # tool_call_id/name/tool_calls 等

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)

    def touch(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1

    def to_message(self) -> dict:
        """转为 OpenAI 风格消息（热记忆中 kind 为 user/assistant/tool/system）。"""
        msg: dict = {"role": self.kind, "content": self.content}
        if self.kind == "assistant" and self.extra.get("tool_calls"):
            msg["tool_calls"] = self.extra["tool_calls"]
        if self.kind == "tool":
            msg["tool_call_id"] = self.extra.get("tool_call_id", "")
            msg["name"] = self.extra.get("name", "")
        return msg


def estimate_tokens(text: str) -> int:
    """粗略 token 估算（生产可换 tiktoken）：按 4 字符 ≈ 1 token 近似。"""
    return max(1, len(text) // 4)


def make_entry(kind: str, content: str, **extra) -> MemoryEntry:
    return MemoryEntry(
        id=uuid.uuid4().hex[:12],
        kind=kind,
        content=content,
        extra=dict(extra),
    )


def priority_score(
    entry: MemoryEntry,
    now: float | None = None,
    w_recency: float = 0.5,
    w_freq: float = 0.3,
    w_importance: float = 0.2,
) -> float:
    """优先级 = 时效性 + 访问频率 + 重要性。低优先级先被压缩/外置。"""
    now = now or time.time()
    age = max(0.0, now - entry.last_accessed)
    recency = math.exp(-age / 3600.0)          # 1 小时半衰期
    freq = math.log1p(entry.access_count)
    importance = 1.0                            # MVP 统一，生产可由模型打分
    return w_recency * recency + w_freq * freq + w_importance * importance
