"""服务层：配置管理。

配置来源优先级：命令行参数 > 环境变量 > 配置文件 > 默认值。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


# provider 预设：环境变量名 / 默认 base_url / 默认模型
PROVIDER_PRESETS = {
    "openai": {"env_key": "OPENAI_API_KEY", "base_url": None, "model": "gpt-4o-mini"},
    "deepseek": {"env_key": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash"},
}


@dataclass
class LLMConfig:
    provider: str = "mock"          # "mock" | "openai" | "deepseek"
    model: str = ""                 # 为空时按 provider 预设解析
    api_key: str = ""               # 按 provider 读对应环境变量
    base_url: str | None = None     # 兼容任意 OpenAI 兼容网关（deepseek/vllm/kimi…）
    temperature: float = 0.2
    extra_body: dict = field(default_factory=dict)  # 透传模型扩展参数（如 thinking/reasoning_effort）

    def resolve(self) -> "LLMConfig":
        preset = PROVIDER_PRESETS.get(self.provider, {})
        env_key = preset.get("env_key", "OPENAI_API_KEY")
        if not self.api_key:
            self.api_key = os.environ.get(env_key, "")
        if not self.base_url:
            self.base_url = os.environ.get("OPENAI_BASE_URL") or preset.get("base_url")
        if not self.model:
            self.model = preset.get("model", "gpt-4o-mini")
        return self


@dataclass
class AgentConfig:
    max_steps: int = 10              # ReAct 循环步数上限
    no_progress_limit: int = 3       # 连续无进展检测阈值
    working_dir: str = "."


@dataclass
class MemoryConfig:
    hot_max_tokens: int = 8000       # 热记忆 token 预算
    warm_max_entries: int = 50
    cold_dir: str = ".aiassist/memory"
    # 五级压缩阈值（token 数）
    truncate_threshold: int = 4000   # L1 单条超长截断
    summarize_threshold: int = 1000  # L2 摘要


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            llm=LLMConfig(**d.get("llm", {})),
            agent=AgentConfig(**d.get("agent", {})),
            memory=MemoryConfig(**d.get("memory", {})),
        )


def load_config(path: str | Path | None = None) -> Config:
    """从配置文件加载，文件不存在则返回默认值。"""
    cfg = Config()
    if path and Path(path).exists():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = Config.from_dict(data)
    cfg.llm.resolve()
    return cfg
