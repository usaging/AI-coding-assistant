"""服务层：组合根（Composition Root）。

在这里把 LLM 客户端、工具注册表、记忆管理器组装成 Agent。
引擎与具体工具/模型彼此不认识，由本模块装配——"调度逻辑与具体工具解耦"的落地。
"""
from __future__ import annotations

from .config import PROVIDER_PRESETS, Config
from .engine import Agent
from .llm.base import LLMClient
from .llm.mock_client import MockLLMClient
from .llm.openai_client import OpenAIClient
from .memory.manager import MemoryManager
from .tools.builtin import ListFilesTool, ReadFileTool, GrepTool, WriteFileTool
from .tools.registry import ToolRegistry


def build_llm(config: Config) -> LLMClient:
    # openai / deepseek 均走 OpenAI 兼容协议，仅 base_url 与默认模型不同
    if config.llm.provider in ("openai", "deepseek"):
        if not config.llm.api_key:
            raise RuntimeError(
                f"使用 {config.llm.provider} provider 需要 API key："
                f"设置环境变量 {PROVIDER_PRESETS[config.llm.provider]['env_key']}"
            )
        return OpenAIClient(
            model=config.llm.model,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature,
            extra_body=config.llm.extra_body,
        )
    return MockLLMClient()  # mock 默认，脚本由调用方注入


def build_registry(working_dir: str) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_many([
        ListFilesTool(working_dir),
        ReadFileTool(working_dir),
        GrepTool(working_dir),
        WriteFileTool(working_dir),
    ])
    return reg


def build_agent(
    config: Config,
    working_dir: str = ".",
    llm: LLMClient | None = None,
) -> Agent:
    llm = llm or build_llm(config)
    registry = build_registry(working_dir)
    memory = MemoryManager(config.memory)
    return Agent(llm=llm, registry=registry, memory=memory, config=config.agent, working_dir=working_dir)
