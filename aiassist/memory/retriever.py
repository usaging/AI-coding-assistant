"""冷记忆检索：关键词命中打分（grep 语义），跨会话复用入口。"""
from __future__ import annotations

import re

from .model import MemoryEntry
from .store import ColdMemoryStore


class ColdRetriever:
    def __init__(self, store: ColdMemoryStore):
        self.store = store

    def search(self, query: str, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        """把自然语言 query 拆成关键词，在冷记忆 Markdown 中 grep，按命中数打分。"""
        terms = [t for t in re.split(r"[\s,，。;；:：]+", query) if len(t) >= 2]
        scored: list[tuple[MemoryEntry, float]] = []
        for p in self.store.files():
            text = p.read_text(encoding="utf-8", errors="replace")
            if not terms:
                score = 0.0
            else:
                low = text.lower()
                score = sum(low.count(t.lower()) for t in terms)
            if score > 0:
                entry = MemoryEntry(
                    id=p.stem, kind="fact",
                    content=self.store._extract_content(text),
                    compression_level=4,
                )
                scored.append((entry, float(score)))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
