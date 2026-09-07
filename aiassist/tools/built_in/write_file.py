"""写入工具：write_file（有副作用，串行执行）。"""
from __future__ import annotations

from pathlib import Path

from ..base import BaseTool, SideEffect, ToolResult
from ._paths import resolve_under


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "创建或覆盖写入文件。路径必须位于 working_dir 内。会覆盖原内容，谨慎使用。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对 working_dir 的文件路径"},
            "content": {"type": "string", "description": "要写入的完整内容"},
        },
        "required": ["path", "content"],
    }
    side_effect = SideEffect.WRITE
    writes_params = ["path"]

    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir

    async def run(self, path: str, content: str) -> ToolResult:
        try:
            p = resolve_under(self.working_dir, path)
            p.parent.mkdir(parents=True, exist_ok=True)
            existed = p.exists()
            p.write_text(content, encoding="utf-8")
            rel = p.relative_to(Path(self.working_dir).resolve())
            verb = "覆盖" if existed else "新建"
            return ToolResult.ok(f"已{verb} {rel}", bytes=len(content.encode("utf-8")))
        except PermissionError as e:
            return ToolResult.fail(str(e))
