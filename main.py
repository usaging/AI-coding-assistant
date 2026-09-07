"""CLI 入口。

用法：
  python main.py "解释一下这个项目的架构" --dir .
  python main.py --demo                          # 离线演示完整 ReAct 循环 + 流式（无需 API key）
  python main.py --demo --plan                   # 离线演示 Plan-and-Execute
  python main.py "..." --provider openai --model gpt-4o-mini
  python main.py "..." --provider deepseek --model deepseek-v4-flash   # 接入 DeepSeek
  python main.py "..." --trace-out .aiassist/trace.json   # 导出执行轨迹
  python main.py --repl                          # 交互式多轮会话
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# 统一输出编码，避免 Windows GBK 控制台下 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from aiassist.bootstrap import build_agent
from aiassist.config import load_config
from aiassist.engine import Event, PlanAndExecuteAgent, Planner, build_trace
from aiassist.llm import LLMResponse, MockLLMClient, ToolCall
from aiassist.service import Session, SessionStore
from aiassist.tokens import estimate_cost

PROJECT_ROOT = Path(__file__).resolve().parent

# 离线 demo 脚本：模拟模型输出 → 先 grep 定位 → 再 read_file 读细节 → 给出结论
DEMO_SCRIPT = [
    LLMResponse(
        tool_calls=[ToolCall(name="grep", arguments={"pattern": "class Agent"})],
        finish_reason="tool_calls",
    ),
    LLMResponse(
        tool_calls=[ToolCall(name="read_file", arguments={"path": "aiassist/engine/agent.py", "limit": 40})],
        finish_reason="tool_calls",
    ),
    LLMResponse(
        content=(
            "已定位核心类 Agent（aiassist/engine/agent.py）。"
            "它实现了 ReAct Query Loop：每轮构建上下文 → 模型推理 → 工具执行（只读并发/写入串行）"
            "→ 结果回填 observation → 终止判断（finish / max_steps / 连续无进展）。"
        ),
        finish_reason="stop",
    ),
]

# 规划器 demo：先产出子任务清单，再逐项执行
PLAN_SCRIPT = [
    LLMResponse(content='["定位核心类 Agent", "阅读 ReAct 循环实现", "总结架构与终止条件"]'),
]


def make_event_printer(stream_text: bool = True, show_tools: bool = True):
    """把引擎事件渲染到终端：流式文本 + 工具进度 + 计划/子任务。"""
    def on_event(ev: Event) -> None:
        if ev.type == "text_delta" and stream_text:
            print(ev.data, end="", flush=True)
        elif ev.type == "tool_result" and show_tools:
            d = ev.data
            status = "OK" if d["success"] else "ERR"
            preview = (d["content"] or d["error"] or "").replace("\n", " ")[:100]
            print(f"\n    [{status}] {d['name']}({d['arguments']}) -> {preview}")
        elif ev.type == "plan":
            print("\n== 计划 ==")
            for i, s in enumerate(ev.data, 1):
                print(f"  {i}. {s}")
        elif ev.type == "subtask":
            d = ev.data
            print(f"\n▶ 子任务 {d['index']}/{d['total']}: {d['title']}")
        elif ev.type == "step_start":
            print(f"\n  [step {ev.data}]", end=" ")
    return on_event


def _print_stats(result, model: str, started_at: float) -> None:
    cost = estimate_cost(model, result.total_tokens_in, result.total_tokens_out)
    print(
        f"\n\n== 统计 ==\n"
        f"  终止原因: {result.terminated_by} | 工具轮数: {len(result.steps)}\n"
        f"  token 输入/输出: {result.total_tokens_in}/{result.total_tokens_out} | "
        f"估算成本: ${cost:.5f} | 耗时: {time.monotonic() - started_at:.2f}s"
    )


async def run_query(config, working_dir: str, query: str, llm=None, trace_out: str | None = None, stream: bool = True):
    started_at = time.monotonic()
    agent = build_agent(config, working_dir, llm=llm)
    on_event = make_event_printer(stream_text=stream)
    result = await agent.run(query, on_event=on_event)
    print("\n\n== 最终答案 ==")
    print(result.final_answer)
    _print_stats(result, config.llm.model, started_at)

    # 会话收尾：温记忆 → L3 聚合 → L4 结构化 → 落冷记忆（跨会话复用）
    facts = agent.memory.finalize()
    if facts:
        print(f"[记忆] 已沉淀 {len(facts)} 条冷记忆到 {config.memory.cold_dir}")

    if trace_out:
        p = build_trace(result, query, config.llm.model, started_at).save(trace_out)
        print(f"[轨迹] 已导出 {p}")

    session = Session(working_dir=working_dir)
    session.add_turn(query, result.final_answer)
    SessionStore().save(session)
    return result


async def run_plan(config, working_dir: str, query: str, planner_llm, agent_llm_factory, trace_out: str | None = None):
    """Plan-and-Execute：先规划子任务，再逐项用独立 Agent 执行。"""
    started_at = time.monotonic()
    planner = Planner(planner_llm)

    def agent_factory():
        return build_agent(config, working_dir, llm=agent_llm_factory())  # 每个子任务独立 Agent

    pa = PlanAndExecuteAgent(planner, agent_factory)
    on_event = make_event_printer(stream_text=True)
    result = await pa.run(query, on_event=on_event)

    print("\n\n== 汇总 ==")
    total_in = total_out = 0
    for r in result["results"]:
        total_in += r["tokens_in"]
        total_out += r["tokens_out"]
        print(f"  • {r['task']}: {r['answer'][:80]}")
    cost = estimate_cost(config.llm.model, total_in, total_out)
    print(f"\n子任务数: {len(result['results'])} | token 输入/输出: {total_in}/{total_out} | 估算成本: ${cost:.5f}")
    return result


async def repl(config, working_dir: str):
    print("== 交互式 REPL（输入 exit 退出，同一 Agent 跨轮复用记忆）==")
    agent = build_agent(config, working_dir)
    session = Session(working_dir=working_dir)
    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            query = ""
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        result = await agent.run(query, on_event=make_event_printer(stream_text=True, show_tools=True))
        print()
        session.add_turn(query, result.final_answer)
    SessionStore().save(session)
    print("\n[会话] 已保存到 .aiassist/sessions/")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 编程助手")
    parser.add_argument("query", nargs="?", default="", help="要执行的任务/问题")
    parser.add_argument("--dir", default=str(PROJECT_ROOT), help="工作目录（默认项目根）")
    parser.add_argument("--provider", default="mock", choices=["mock", "openai", "deepseek"], help="LLM 提供方")
    parser.add_argument("--model", default=None, help="模型名（默认按 provider：openai→gpt-4o-mini，deepseek→deepseek-v4-flash）")
    parser.add_argument("--base-url", default=None, help="覆盖 API base_url（兼容任意 OpenAI 兼容网关）")
    parser.add_argument("--thinking", action="store_true", help="启用 DeepSeek thinking 模式（默认关闭）")
    parser.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"], help="DeepSeek reasoning_effort（low/medium/high）")
    parser.add_argument("--demo", action="store_true", help="离线演示完整 ReAct 循环")
    parser.add_argument("--plan", action="store_true", help="使用 Plan-and-Execute 规划器")
    parser.add_argument("--trace-out", default=None, help="导出执行轨迹 JSON 到指定路径")
    parser.add_argument("--repl", action="store_true", help="交互式多轮会话")
    args = parser.parse_args()

    config = load_config()
    config.llm.provider = args.provider
    if args.model:
        config.llm.model = args.model
    if args.base_url:
        config.llm.base_url = args.base_url
    extra_body: dict = {}
    if args.thinking:
        extra_body["thinking"] = {"type": "enabled"}
    if args.reasoning_effort:
        extra_body["reasoning_effort"] = args.reasoning_effort
    if extra_body:
        config.llm.extra_body = extra_body
    config.llm.resolve()

    if args.repl:
        asyncio.run(repl(config, args.dir))
        return

    if args.demo:
        if args.plan:
            print("== 离线 demo：Plan-and-Execute（规划 → 逐项执行）==\n")
            planner_llm = MockLLMClient(PLAN_SCRIPT)
            sub_scripts = iter([
                [LLMResponse(tool_calls=[ToolCall(name="grep", arguments={"pattern": "class Agent"})]),
                 LLMResponse(content="已定位核心类 Agent。")],
                [LLMResponse(tool_calls=[ToolCall(name="read_file", arguments={"path": "aiassist/engine/agent.py", "limit": 30})]),
                 LLMResponse(content="已阅读 ReAct 循环实现。")],
                [LLMResponse(content="总结：三层架构 + ReAct 循环 + 四类终止条件。")],
            ])

            def sub_llm():
                return MockLLMClient(next(sub_scripts))

            asyncio.run(run_plan(config, args.dir, "介绍这个项目的架构", planner_llm, sub_llm))
        else:
            print("== 离线 demo：ReAct Query Loop（流式 + 工具进度）==\n")
            llm = MockLLMClient(DEMO_SCRIPT)
            asyncio.run(run_query(config, args.dir, "介绍一下项目里 Agent 的实现", llm=llm, trace_out=args.trace_out))
        return

    if not args.query:
        parser.error("请提供任务描述，或用 --demo / --repl 查看演示")

    if args.plan:
        from aiassist.bootstrap import build_llm
        asyncio.run(run_plan(config, args.dir, args.query, build_llm(config), lambda: build_llm(config), args.trace_out))
    else:
        asyncio.run(run_query(config, args.dir, args.query, trace_out=args.trace_out))


if __name__ == "__main__":
    main()
