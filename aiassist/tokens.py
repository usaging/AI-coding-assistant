"""Token 计数与成本估算。

- 精确计数：优先使用 tiktoken（cl100k_base），未安装则回退启发式。
- 成本：按模型定价表（美元/1k tokens）估算，用于可观测性与评估。
"""
from __future__ import annotations

import re

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except Exception:  # tiktoken 为可选依赖
    _ENC = None
    _HAS_TIKTOKEN = False

_CJK = re.compile(r"[\u4e00-\u9fff]")


def count_tokens(text: str) -> int:
    if _HAS_TIKTOKEN:
        return len(_ENC.encode(text))
    # 启发式：中文 ≈ 1 字 1 token，其余 ≈ 4 字符 1 token
    cjk = len(_CJK.findall(text))
    other = max(0, len(text) - cjk)
    return max(1, cjk + other // 4)


# 默认定价（美元 / 1k tokens）
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"in": 0.00015, "out": 0.0006},
    "gpt-4o": {"in": 0.0025, "out": 0.01},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p = DEFAULT_PRICES.get(model, {"in": 0.00015, "out": 0.0006})
    return (tokens_in * p["in"] + tokens_out * p["out"]) / 1000.0
