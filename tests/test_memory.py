"""验证记忆系统：五级压缩、超预算淘汰、冷记忆落盘与 grep 召回。"""
from __future__ import annotations

import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiassist.config import MemoryConfig
from aiassist.memory import MemoryManager, MemoryTier, make_entry

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def ws_tmpdir():
    """workspace 内临时目录（避免 %TEMP% 及 mkdtemp 被沙箱限制）。"""
    d = ROOT / ("tmp-test-" + uuid.uuid4().hex[:8])
    d.mkdir(parents=True, exist_ok=False)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_escalation_reduces_tokens():
    cfg = MemoryConfig()
    m = MemoryManager(cfg)
    entry = make_entry("tool", "x" * 10000)
    before = entry.tokens

    m.compressor.escalate(entry)          # L1 截断
    assert entry.compression_level == 1
    assert entry.tokens < before

    m.compressor.escalate(entry)          # L2 摘要
    assert entry.compression_level == 2
    assert entry.kind == "summary"


def test_evict_when_over_budget():
    cfg = MemoryConfig(hot_max_tokens=150)  # 很小的预算
    m = MemoryManager(cfg)
    m.add_message("system", "系统提示")
    for i in range(6):
        m.add_message("user", "数据 " + "word" * 300 + f" {i}")

    archived = m.evict_if_needed()
    # 要么已压到预算内，要么只剩最小保留条数
    assert m.hot_tokens() <= cfg.hot_max_tokens or len(m.hot) <= 4
    # 淘汰出的条目被降级到温记忆
    for e in archived:
        assert e.tier == MemoryTier.WARM


def test_finalize_and_recall():
    with ws_tmpdir() as tmp:
        cfg = MemoryConfig(cold_dir=str(tmp / "memory"))
        m = MemoryManager(cfg)
        m.warm.append(make_entry("summary", "并发执行器设计：只读工具并行，写入工具串行保序。"))
        m.warm.append(make_entry("summary", "记忆分热温冷三层，冷记忆用 grep 检索 Markdown。"))

        facts = m.finalize()
        assert len(facts) == 1
        assert len(m.store.files()) == 1  # 落盘为 Markdown

        # 跨会话召回：相关关键词能命中
        hits = m.recall("只读 并行 写入 串行")
        assert hits, "应能召回冷记忆"
        assert "并行" in hits[0][0].content


if __name__ == "__main__":
    test_escalation_reduces_tokens()
    test_evict_when_over_budget()
    test_finalize_and_recall()
    print("test_memory OK")
