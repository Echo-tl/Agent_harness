
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from runtime.context import get_run_context

@tool("task")
async def task_tool(research_topic: str) -> str:
    """委派研究任务给 Researcher 子 Agent"""

    get_run_context().push_progress(f"开始研究子任务：{research_topic[:60]}")

    # Lazy import to break circular dependency:
    # graph → supervisor → executor → graph
    from graph import build_research_subgraph

    # ① 创建 Researcher
    researcher = build_research_subgraph()

    # ②构建初始 state
    initial_state = {
        "messages": [
            HumanMessage(content=research_topic)
        ],  # 新 Researcher，空对话
        "research_topic": research_topic,
        "tool_call_iterations": 0,
    }

    # ③ 执行 Researcher 子图
    result = await researcher.ainvoke(initial_state)

    # ④ 返回 compressed_research
    return result["compressed_research"]
