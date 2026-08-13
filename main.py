"""
Mini Research Agent —— 入口文件

运行方式：
  cd mini_deer-flow_agent
  python main.py
"""

from dotenv import load_dotenv
load_dotenv()

import uuid
import asyncio
from langgraph.types import Command
from graph import build_parent_graph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from runtime.context import RunContext, run_scope
from config import CONFIG

async def main():
    """主函数：构建图、运行、展示结果"""

    # 1. 构建图

    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        graph = build_parent_graph(checkpointer=checkpointer)

        # 会话 ID
        thread_id = str(uuid.uuid4())   # 每次运行生成唯一 ID，不复用旧状态
        config = {"configurable": {"thread_id": thread_id}}
        timeout = float(CONFIG["limits"].get("timeout_seconds", 600))

        # 每次运行独立的 RunContext（cost / visited_urls / evidence 按 thread_id 隔离）
        ctx = RunContext(thread_id=thread_id)

        print("Mini Research Agent V9 (Async Parallel)")
        print("   输入问题开始研究，输入 'exit' 退出")

        first_question = input("\n你的问题: ")
        if first_question.lower() == "exit":
            return

        # 2. 准备输入
        from langchain_core.messages import HumanMessage
        initial_state = {
            "messages": [HumanMessage(content=first_question)],
            "research_brief": "",
            "final_report": "",
            "total_cost": 0.0,
            "citation_stats": {},
        }

        # 3. 异步执行图（受总超时限制）
        print("\n" + "=" * 60)
        print("[Start] executing Graph...")
        print("=" * 60)

        last_state = None
        with run_scope(ctx):
            try:
                async with asyncio.timeout(timeout):
                    async for chunk in graph.astream(initial_state, config, stream_mode="values"):
                        print(f"当前 messages 数量: {len(chunk.get('messages', []))}")
                        last_state = chunk   # 保存最后一次
            except TimeoutError:
                print(f"\n[Error] 研究超时（超过 {timeout} 秒），已停止。")
                return

            # 检查是否被 interrupt
            state = await graph.aget_state(config)

            # state.next 不为空 →说明图被 interrupt 了，等着 resume
            while state.next:
                # 从 state 里取出 interrupt 的信息（LLM 追问的内容）
                # 展示给用户，等用户输入
                interrupt_msg = ""
                for task in state.tasks:
                    if task.interrupts:
                        interrupt_msg = task.interrupts[0].value
                        break

                user_input = input(f"\n{interrupt_msg}\n> ")

                # 用 resume 恢复 →流式看后续进度
                async with asyncio.timeout(timeout):
                    async for chunk in graph.astream(Command(resume=user_input), config, stream_mode="values"):
                        last_state = chunk

                state = await graph.aget_state(config)

        print(f"\n[Answer]:\n{last_state.get('final_report', 'No report generated')}")

        # 后续追问
        while True:
            follow_up = input("\n追问（或 exit 退出）: ")
            if follow_up.lower() == "exit":
                break

            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=follow_up)]},
                config,
            )


if __name__ == "__main__":
    asyncio.run(main())
