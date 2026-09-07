"""五级压缩策略 L0 → L4。

- L0 原始：原样保留。
- L1 截断：超长内容结构化截断（留头尾），关键信息无损。
- L2 摘要：单条压缩为 1-2 句（默认启发式抽取，可注入 LLM summarizer）。
- L3 聚合：多条同主题条目合并为一条任务 digest（移入温记忆）。
- L4 结构化：抽取可复用事实，落冷记忆 Markdown（跨会话复用）。

按上下文压力逐级升级，低优先级条目先被升级。
"""
from __future__ import annotations

import re
import time
import uuid
from collections import Counter

from .model import MemoryEntry, MemoryTier

# 英文标识符 或 中文连续片段
_LATIN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CHINESE = re.compile(r"[\u4e00-\u9fff]+")


def _keywords(content: str, top_n: int = 5) -> list[str]:
    """启发式关键词：英文标识符 + 中文二字词（bigram），按频次取前 top_n。"""
    latin = _LATIN.findall(content)
    cn_terms: list[str] = []
    for seg in _CHINESE.findall(content):
        cn_terms.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return [w for w, _ in Counter(latin + cn_terms).most_common(top_n)]


class Compressor:
    HEAD = 1500
    TAIL = 400

    def __init__(self, summarizer=None):
        # summarizer: callable(content)->str，生产注入 LLM 摘要；默认启发式抽取
        self.summarizer = summarizer

    def truncate(self, content: str) -> str:
        """L1：截断超长内容，保留头部 + 尾部。"""
        if len(content) <= self.HEAD + self.TAIL:
            return content
        return content[: self.HEAD] + "\n...[L1 截断]...\n" + content[-self.TAIL:]

    def summarize(self, content: str) -> str:
        """L2：摘要。有 LLM summarizer 用之，否则启发式（前两行 + 高频关键词）。"""
        if self.summarizer:
            return self.summarizer(content)
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        head = " ".join(lines[:2])[:200]
        kw = _keywords(content)
        return f"{head}" + (f" | 关键词: {', '.join(kw)}" if kw else "")

    def escalate(self, entry: MemoryEntry) -> MemoryEntry:
        """就地提升一级：L0→L1→L2，减小体积。"""
        if entry.compression_level == 0:
            entry.content = self.truncate(entry.content)
            entry.compression_level = 1
        elif entry.compression_level == 1:
            entry.content = self.summarize(entry.content)
            entry.compression_level = 2
            entry.kind = "summary"
        return entry

    def aggregate(self, entries: list[MemoryEntry], title: str = "任务摘要") -> MemoryEntry:
        """L3：聚合多条为一条 digest（移入温记忆）。"""
        merged = "\n".join(f"- {e.content}" for e in entries)
        return MemoryEntry(
            id=uuid.uuid4().hex[:12],
            kind="summary",
            content=f"[{title}]\n{merged}",
            tier=MemoryTier.WARM,
            compression_level=3,
        )

    def to_structured_fact(self, entry: MemoryEntry) -> MemoryEntry:
        """L4：抽取结构化事实（要点 + 标签），落冷记忆。"""
        summary = self.summarize(entry.content)
        tags = " ".join(_keywords(entry.content, top_n=8))
        body = f"要点: {summary}\n标签: {tags}"
        return MemoryEntry(
            id=uuid.uuid4().hex[:12],
            kind="fact",
            content=body,
            tier=MemoryTier.COLD,
            compression_level=4,
            last_accessed=time.time(),
        )
