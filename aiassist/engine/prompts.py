"""System prompt 构造：角色定义 + 工作目录 + 工具 schema + 执行规范。"""
from __future__ import annotations


SYSTEM_TEMPLATE = """你是 AI 编程助手，面向代码仓库执行开发与理解任务。

工作目录: {working_dir}

执行规范：
1. 先检索再修改：不确定文件位置时，先用 grep / list_files / read_file 探查。
2. 写操作前确认目标文件存在且路径正确。
3. 工具失败会以错误信息回填，请根据错误调整策略（重试/换工具/放弃）。
4. 任务完成后直接给出最终结论，不要继续调用工具。

可用工具（JSON Schema 见下方 function definitions，请按 schema 传参）：
{tools}
"""


def build_system_prompt(working_dir: str, tool_schemas: list[dict]) -> str:
    tools_desc = "\n".join(
        f"- {s['function']['name']}: {s['function']['description']}"
        for s in tool_schemas
    )
    return SYSTEM_TEMPLATE.format(working_dir=working_dir, tools=tools_desc)
