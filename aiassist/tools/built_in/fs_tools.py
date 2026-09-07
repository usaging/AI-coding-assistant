"""只读文件系统工具：list_files / read_file / grep。"""
from __future__ import annotations

import re
from pathlib import Path

from ..base import BaseTool, SideEffect, ToolResult
from ._paths import resolve_under


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "列出指定目录下的文件和子目录，支持限制深度。用于了解仓库结构。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对 working_dir 的目录路径，默认 '.'"},
            "depth": {"type": "integer", "description": "递归深度，默认 2"},
        },
        "required": [],
    }
    side_effect = SideEffect.READ
    reads_params = ["path"]

    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir

    async def run(self, path: str = ".", depth: int = 2) -> ToolResult:
        try:
            root = resolve_under(self.working_dir, path)
            if not root.is_dir():
                return ToolResult.fail(f"不是目录: {path}")
            lines = self._walk(root, depth, 0)
            return ToolResult.ok("\n".join(lines) or "(空目录)")
        except PermissionError as e:
            return ToolResult.fail(str(e))

    def _walk(self, dir_: Path, depth: int, level: int) -> list[str]:
        lines: list[str] = []
        entries = sorted(dir_.iterdir(), key=lambda p: (p.is_file(), p.name))
        for p in entries:
            if p.name.startswith("."):  # 跳过隐藏文件/目录（如 .git/.aiassist）
                continue
            indent = "  " * level
            if p.is_dir():
                lines.append(f"{indent}{p.name}/")
                if level < depth:
                    lines.extend(self._walk(p, depth, level + 1))
            else:
                lines.append(f"{indent}{p.name}")
        return lines


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "读取文件内容，支持分页（offset/limit）。文件过大会截断。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对 working_dir 的文件路径"},
            "offset": {"type": "integer", "description": "起始行（1 开始），默认 1"},
            "limit": {"type": "integer", "description": "读取行数，默认 200"},
        },
        "required": ["path"],
    }
    side_effect = SideEffect.READ
    reads_params = ["path"]

    MAX_LINES = 2000

    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir

    async def run(self, path: str, offset: int = 1, limit: int = 200) -> ToolResult:
        try:
            p = resolve_under(self.working_dir, path)
            if not p.is_file():
                return ToolResult.fail(f"文件不存在: {path}")
            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            offset = max(1, offset)
            limit = max(1, min(limit, self.MAX_LINES))
            chunk = lines[offset - 1: offset - 1 + limit]
            content = "\n".join(chunk)
            meta = {"total_lines": len(lines), "offset": offset, "shown": len(chunk)}
            return ToolResult.ok(content, **meta)
        except PermissionError as e:
            return ToolResult.fail(str(e))


class GrepTool(BaseTool):
    name = "grep"
    description = "在 working_dir 下按关键词/正则检索文件内容，返回 file:line:content。用于代码搜索和记忆检索。"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "关键词或正则表达式"},
            "path": {"type": "string", "description": "限定搜索的目录，默认 '.'"},
            "max_matches": {"type": "integer", "description": "最大匹配数，默认 50"},
        },
        "required": ["pattern"],
    }
    side_effect = SideEffect.READ
    reads_params = ["path"]

    DEFAULT_IGNORE = {".git", ".aiassist", "node_modules", "__pycache__", ".venv"}

    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir

    async def run(self, pattern: str, path: str = ".", max_matches: int = 50) -> ToolResult:
        try:
            regex = re.compile(pattern)
            root = resolve_under(self.working_dir, path)
            matches: list[str] = []
            for p in self._iter_files(root):
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for lineno, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        matches.append(f"{p.name}:{lineno}:{line.strip()}")
                        if len(matches) >= max_matches:
                            return ToolResult.ok("\n".join(matches), count=len(matches))
            return ToolResult.ok("\n".join(matches) or "(无匹配)", count=len(matches))
        except (PermissionError, re.error) as e:
            return ToolResult.fail(str(e))

    def _iter_files(self, root: Path):
        for p in root.rglob("*"):
            if any(part in self.DEFAULT_IGNORE for part in p.parts):
                continue
            if p.is_file():
                yield p
