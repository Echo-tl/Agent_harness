
from nodes.report import final_report
from nodes.planner import planner_node
from nodes.compress import compress_node

from langgraph.graph import StateGraph, END, START
from state import ResearcherState, ResearcherOutputState,  MultiAgentState

from tools import search, rag_search, memory_search
from config import CONFIG
from llm_config import llm
from prompts import SYSTEM_PROMPT

from langchain.agents import create_agent
from middleware.summarization import SummarizationMiddleware
from middleware.loop_detection import LoopDetectionMiddleware
from middleware.tool_error_handling import ToolErrorHandlingMiddleware
from middleware.dynamic_context import DynamicContextMiddleware
from middleware.token_budget import TokenBudgetMiddleware
from middleware.run_guard import RunGuardMiddleware
from middleware import enabled_middleware
from sandbox.tools import read_file, write_file, ls, bash
from tools.skill_loader import use_skill
from mini_mcp.loader import get_mcp_tools
from skills.storage import load_enabled_skills, build_skills_index

# 渐进式披露：上下文只放 skill 轻量索引（name + description），
# 完整方法论由 use_skill 工具按需加载
_SKILLS_INDEX = build_skills_index(load_enabled_skills())
_SKILL_HINT = (
    "\n\n" + _SKILLS_INDEX +
    "\n\n可用技能按需加载（渐进式披露）：当任务匹配某个 skill 时，调用 use_skill('<name>') 获取其完整方法论再执行。"
    if _SKILLS_INDEX else ""
)

def build_research_subgraph():
    """构建 Researcher 子图：接收 research_topic，输出 compressed_research"""

    # 加载 MCP 工具
    mcp_tools = get_mcp_tools()

    limits = CONFIG["limits"]

    # 1. 用 create_agent 创建 agent（内部有 ReAct）
    agent = create_agent(
        model=llm,
        tools=[search, rag_search, memory_search, read_file, write_file, ls, bash, use_skill] + mcp_tools,
        system_prompt=SYSTEM_PROMPT + _SKILL_HINT,
        state_schema=ResearcherState,
        middleware=enabled_middleware([
            SummarizationMiddleware(name="Researcher", max_tokens=10000),
            LoopDetectionMiddleware(name="researcher_loop"),
            ToolErrorHandlingMiddleware(),
            DynamicContextMiddleware(),
            TokenBudgetMiddleware(name="researcher_token", max_tokens=10000),
            # 真正生效的迭代 + 成本上限
            # max_researcher_tool_calls 按"搜索轮次"计数（search / rag_search），
            # 其它工具调用由 LoopDetection + TokenBudget + 总超时兜底
            RunGuardMiddleware(
                counter_key="researcher",
                tool_names=["search", "rag_search"],
                max_calls=limits["max_researcher_tool_calls"],
                max_cost=limits["max_cost"],
            ),
        ]),
    )

    # 2. 包一层图：agent →compress →END
    builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
    builder.add_node("agent", agent)
    builder.add_node("compress", compress_node)

    builder.add_edge(START, "agent")
    builder.add_edge("agent", "compress")   # agent 跑完自动走 compress
    builder.add_edge("compress", END)

    return builder.compile()

from nodes.supervisor import get_supervisor_agent
def build_parent_graph(checkpointer=None):
    """构建 父图：接收 用户问题，输出"""

    builder = StateGraph(MultiAgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", get_supervisor_agent())
    builder.add_node("final_report", final_report)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_edge("supervisor", "final_report")
    builder.add_edge("final_report", END)

    # 对话的 checkpoint（包括 interrupt 状态）存入 checkpoints.db 这个 sqlite 文件。程序关掉后，下次用同一个 thread_id可以 resume
    return builder.compile(checkpointer=checkpointer)
