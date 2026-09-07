"""冷记忆持久化：Markdown 文件（一个条目一个文件）。

选 Markdown 而非数据库的理由：人类可读可 diff、可直接被 grep/git 生态处理、MVP 无并发写事务需求。
"""
from __future__ import annotations

from pathlib import Path

from .model import MemoryEntry, MemoryTier


class ColdMemoryStore:
    def __init__(self, cold_dir: str | Path):
        self.dir = Path(cold_dir)

    def save(self, entry: MemoryEntry) -> Path:
        """把冷记忆条目写为 Markdown 文件。"""
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{entry.id}.md"
        md = (
            f"## Fact: {entry.id}\n"
            f"kind: {entry.kind}\n"
            f"priority: {entry.priority:.3f}\n"
            f"updated_at: {entry.last_accessed}\n"
            f"---\n"
            f"{entry.content}\n"
        )
        path.write_text(md, encoding="utf-8")
        return path

    def files(self) -> list[Path]:
        if not self.dir.exists():
            return []
        return sorted(self.dir.glob("*.md"))

    def load_all(self) -> list[MemoryEntry]:
        """读回所有条目（MVP：还原 content 用于检索/展示）。"""
        entries: list[MemoryEntry] = []
        for p in self.files():
            text = p.read_text(encoding="utf-8", errors="replace")
            content = self._extract_content(text)
            entries.append(
                MemoryEntry(
                    id=p.stem, kind="fact", content=content,
                    tier=MemoryTier.COLD, compression_level=4,
                )
            )
        return entries

    @staticmethod
    def _extract_content(md: str) -> str:
        if "---" in md:
            return md.split("---", 1)[1].strip()
        return md
