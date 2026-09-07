# AI 编程助手
面向代码仓库开发场景的智能 Agent，参考 Claude Code 的交互与执行机制。

## 快速开始

```bash
cd ai-coding-assistant

# 1) 离线演示 ReAct 循环（流式 + 工具进度，无需 API key）
python main.py --demo

# 2) 离线演示 Plan-and-Execute（规划 → 逐项执行）
python main.py --demo --plan

# 3) 交互式多轮 REPL（同一 Agent 跨轮复用记忆）
python main.py --repl

# 4) 执行任务并导出执行轨迹 JSON
python main.py "解释一下 agent.py 里的循环逻辑" --trace-out .aiassist/trace.json

# 5) 接入真实模型（OpenAI）
pip install openai
set OPENAI_API_KEY=sk-xxx
python main.py "找出所有定义了 Tool 的类" --provider openai --model gpt-4o-mini

# 6) 接入 DeepSeek（OpenAI 兼容，base_url=https://api.deepseek.com）
set DEEPSEEK_API_KEY=sk-xxx
python main.py "解释一下 agent.py 的循环逻辑" --provider deepseek                 # 默认 deepseek-v4-flash
python main.py "..." --provider deepseek --model deepseek-v4-pro                   # 更高推理质量
python main.py "..." --provider deepseek --thinking --reasoning-effort high        # 启用思考模式

# 7) 运行评估 harness（mock 确定性任务，统计成功率/成本）
python eval/evaluate.py

# 8) 运行测试
python -m pytest -q
# 或逐个独立运行：
python tests/test_executor.py
python tests/test_executor_dag.py
python tests/test_memory.py
python tests/test_agent_mock.py
python tests/test_planner.py
```

## 三层架构速览

| 层 | 职责 | 关键类 |
|---|---|---|
| 引擎层 | 任务编排、Query Loop、规划、上下文管理、终止判断 | `Agent` / `Planner` / `PlanAndExecuteAgent` |
| 工具层 | 统一工具接口、注册、参数校验、资源级依赖并发执行 | `BaseTool` / `ToolRegistry` / `ToolExecutor` |
| 服务层 | 会话、配置、持久化、组合根 | `Session` / `Config` / `bootstrap` |

引擎只依赖 `LLMClient`（模型抽象）+ `ToolRegistry`（工具集）+ `MemoryManager`（记忆）三个抽象，不 import 任何具体工具。

新增工具 = 写一个 `BaseTool` 子类（声明 `reads_params`/`writes_params`）+ `registry.register(...)`，引擎零改动。


## 目录结构

```
ai-coding-assistant/
├── main.py                  # CLI 入口（demo / plan / repl / trace）
├── conftest.py              # pytest 根路径配置
├── requirements.txt
├── docs/
│   ├── DESIGN.md            # 完整设计文档
│   └── INTERVIEW.md         # 面试话术 + 抗追问 Q&A
├── eval/
│   └── evaluate.py          # 评估 harness（任务集 + 评分）
├── aiassist/
│   ├── config.py            # 服务层：配置
│   ├── bootstrap.py         # 服务层：组合根（装配 LLM/工具/记忆 → Agent）
│   ├── tokens.py            # token 计数 + 成本估算
│   ├── engine/              # 引擎层
│   │   ├── agent.py         #   ReAct Query Loop（流式 + 计时/token）
│   │   ├── planner.py       #   Plan-and-Execute 规划器
│   │   ├── events.py        #   事件模型（text_delta/tool_result/plan/…）
│   │   ├── trace.py         #   执行轨迹 + JSON 导出
│   │   ├── prompts.py       #   system prompt 构造
│   │   └── types.py         #   AgentStep / AgentResult
│   ├── llm/                 # 模型交互抽象
│   │   ├── base.py          #   LLMClient 接口（流式 on_delta）
│   │   ├── openai_client.py #   function calling + stream
│   │   └── mock_client.py   #   离线/测试用脚本化 mock
│   ├── tools/               # 工具层
│   │   ├── base.py          #   BaseTool / ToolResult / SideEffect / 资源声明
│   │   ├── registry.py      #   注册表（解耦关键）
│   │   ├── executor.py      #   资源级依赖 DAG + 只读并发/写入串行
│   │   ├── validator.py     #   参数 JSON Schema 校验
│   │   └── builtin/         #   list_files / read_file / grep / write_file
│   ├── memory/              # 记忆系统
│   │   ├── model.py         #   MemoryEntry / 热温冷 / 优先级
│   │   ├── compressor.py    #   五级压缩 L0-L4
│   │   ├── store.py         #   Markdown 持久化
│   │   ├── retriever.py     #   grep 检索
│   │   └── manager.py       #   编排（超预算淘汰 / 会话收尾）
│   └── service/             # 服务层：会话持久化
│       └── session.py
└── tests/                   # 单元 + 端到端（mock）测试
```
