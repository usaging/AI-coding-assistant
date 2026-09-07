"""路径安全：所有文件工具统一走这里，防止越界访问。"""
from __future__ import annotations

from pathlib import Path


def resolve_under(working_dir: str | Path, path: str | Path) -> Path:
    """将 path 解析到 working_dir 下，拒绝越界。

    - 绝对路径：仅当其确实位于 working_dir 内才允许。
    - 相对路径：resolve 后校验前缀，拒绝 `../` 逃逸。
    """
    root = Path(working_dir).expanduser().resolve()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    # 校验是否在 root 内（root 本身或其后代）
    if p != root and root not in p.parents:
        raise PermissionError(f"路径越界，禁止访问 working_dir 之外: {path}")
    return p
