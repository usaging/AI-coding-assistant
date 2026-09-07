"""工具参数 JSON Schema 校验。

在执行前校验参数（必填、类型），把"缺参/错参"转成可读错误回填给模型，
而不是等工具内部 KeyError——让模型能根据错误自我修正。
"""
from __future__ import annotations

from typing import Any

_TYPE_CHECKS = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}

_TYPE_NAMES = {
    "string": "字符串", "integer": "整数", "number": "数字",
    "boolean": "布尔", "array": "数组", "object": "对象",
}


def validate_arguments(schema: dict, arguments: dict[str, Any]) -> str | None:
    """校验通过返回 None，否则返回可读错误信息。"""
    if schema.get("type") != "object":
        return None  # 非对象 schema 不校验
    props: dict = schema.get("properties", {})
    required: list = schema.get("required", [])

    for key in required:
        if key not in arguments:
            return f"缺少必填参数 '{key}'"

    for key, value in arguments.items():
        if key not in props:
            continue  # 多余参数由模型/模型 API 忽略，这里宽容处理
        ptype = props[key].get("type")
        if ptype is None:
            continue
        expected = _TYPE_CHECKS.get(ptype)
        if expected is None:
            continue
        if not isinstance(value, expected):
            return f"参数 '{key}' 应为{_TYPE_NAMES.get(ptype, ptype)}，实际为 {type(value).__name__}"

    return None
