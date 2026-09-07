"""AI 编程助手（MVP 骨架）。

三层架构：
- engine/  引擎层：Agent 编排、Query Loop、上下文管理
- tools/   工具层：注册表、执行器、内置工具、副作用并发
- service/ 服务层：会话、配置、持久化（组合根）
- memory/  记忆：热/温/冷 + 五级压缩 + grep 检索
"""

__version__ = "0.1.0"
