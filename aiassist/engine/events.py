"""引擎事件：供 CLI / 前端订阅，实现流式与进度展示，解耦引擎与展示层。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Event:
    type: str
    data: Any = None


# 事件类型约定：
#   ("text_delta", str)         流式文本增量
#   ("step_start", int)         第 N 步开始
#   ("tool_result", dict)       单个工具结果 {name, success, content, error}
#   ("final", str)              最终答案
#   ("plan", list[str])         规划器产出的子任务列表
#   ("subtask", dict)           子任务开始 {index, total, title}
