"""记忆管理器：协调热/温/冷三层 + 五级压缩触发。"""
from __future__ import annotations

from ..config import MemoryConfig
from .compressor import Compressor
from .model import MemoryEntry, MemoryTier, make_entry, priority_score
from .retriever import ColdRetriever
from .store import ColdMemoryStore


class MemoryManager:
    def __init__(self, config: MemoryConfig, summarizer=None):
        self.config = config
        self.hot: list[MemoryEntry] = []
        self.warm: list[MemoryEntry] = []
        self.compressor = Compressor(summarizer)
        self.store = ColdMemoryStore(config.cold_dir)
        self.retriever = ColdRetriever(self.store)

    # ---- 写入 ----
    def add(self, entry: MemoryEntry) -> None:
        self.hot.append(entry)

    def add_message(self, role: str, content: str, **extra) -> None:
        self.add(make_entry(role, content, **extra))

    # ---- 读取 ----
    def build_messages(self) -> list[dict]:
        """热记忆 → LLM 消息序列。"""
        return [e.to_message() for e in self.hot]

    def hot_tokens(self) -> int:
        return sum(e.tokens for e in self.hot)

    # ---- 压缩触发 ----
    def evict_if_needed(self, keep_min: int = 4) -> list[MemoryEntry]:
        """上下文超预算时：低优先级条目先升级压缩级别，L2 以上外置到温记忆。"""
        archived: list[MemoryEntry] = []
        while self.hot_tokens() > self.config.hot_max_tokens and len(self.hot) > keep_min:
            candidates = [e for e in self.hot if e.kind != "system"]
            if len(candidates) <= 1:  # 只剩最后一条，无法再压
                break
            candidates = candidates[:-1]  # 保留最新一条上下文
            victim = min(candidates, key=lambda e: priority_score(e))
            if victim.compression_level < 2:
                self.compressor.escalate(victim)
            else:
                self.hot.remove(victim)
                victim.tier = MemoryTier.WARM
                self.warm.append(victim)
                archived.append(victim)
        return archived

    # ---- 跨会话 ----
    def recall(self, query: str, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        """冷记忆检索，按相关度回填。"""
        return self.retriever.search(query, top_k)

    def finalize(self) -> list[MemoryEntry]:
        """会话收尾：温记忆 → L3 聚合 → L4 结构化 → 落冷记忆 Markdown。"""
        if not self.warm:
            return []
        digest = self.compressor.aggregate(self.warm, title="会话沉淀")  # L3
        fact = self.compressor.to_structured_fact(digest)                 # L4
        self.store.save(fact)
        self.warm.clear()
        return [fact]
