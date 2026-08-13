"""Supervisor 节点 ——研究总指挥，决定启动哪个 Researcher 或结束研究"""

from state.models import ResearchComplete
from llm_config import llm as supervisor_llm
from prompts import SUPERVISOR_SYSTEM_PROMPT

 # ①task_tool 替代了 ConductResearch（已经从 subagents/executor.py import）
from subagents.executor import task_tool
from langchain.agents import create_agent

from middleware.summarization import SummarizationMiddleware
from middleware.loop_detection import LoopDetectionMiddleware
from middleware.tool_error_handling import ToolErrorHandlingMiddleware
from middleware.dynamic_context import DynamicContextMiddleware
from middleware.clarification import ClarificationMiddleware
from middleware.token_budget import TokenBudgetMiddleware
from middleware.run_guard import RunGuardMiddleware
from middleware import enabled_middleware

from langchain_core.tools import tool
from config import CONFIG
from tools.skill_loader import use_skill
from skills.storage import load_enabled_skills, build_skills_index

@tool("ask_clarification")
def ask_clarification_tool(question: str) -> str:
    """向用户追问澄清问题。当用户问题模糊时可以调用。"""
    return "middleware 拦截了此调用"

# 渐进式披露：Supervisor 上下文只放 skill 轻量索引，完整方法论由 use_skill 按需加载
_SKILLS_INDEX = build_skills_index(load_enabled_skills())
_SKILL_HINT = (
    "\n\n" + _SKILLS_INDEX +
    "\n\n可用技能按需加载（渐进式披露）：当任务匹配某个 skill 时，调用 use_skill('<name>') 获取其完整方法论再执行。"
    if _SKILLS_INDEX else ""
)


def _build_supervisor_agent():
    """构建 Supervisor agent（读最新 CONFIG，评测 A/B 时据此重建）。"""
    return create_agent(
        model=supervisor_llm,
        tools=[task_tool, ResearchComplete, ask_clarification_tool, use_skill],
        system_prompt=SUPERVISOR_SYSTEM_PROMPT + _SKILL_HINT,
        middleware=enabled_middleware([
            SummarizationMiddleware(name="Supervisor", max_tokens=30000),
            ToolErrorHandlingMiddleware(),
            DynamicContextMiddleware(),
            ClarificationMiddleware(),
            TokenBudgetMiddleware(name="supervisor_token", max_tokens=30000),
            # LoopDetection 不放在 Supervisor —— task() 被多次调用是正常行为
            # 真正生效的迭代上限：只统计 task 调用（max_supervisor_iterations）
            RunGuardMiddleware(
                counter_key="supervisor_task",
                tool_names=["task"],
                max_calls=CONFIG["limits"]["max_supervisor_iterations"],
                max_cost=CONFIG["limits"]["max_cost"],
            ),
        ]),
    )


_supervisor_agent = None


def get_supervisor_agent():
    """惰性构建并缓存 Supervisor agent。"""
    global _supervisor_agent
    if _supervisor_agent is None:
        _supervisor_agent = _build_supervisor_agent()
    return _supervisor_agent


def reset_supervisor_agent():
    """清除缓存，下次 get_supervisor_agent() 按最新 CONFIG 重建（评测 A/B 用）。"""
    global _supervisor_agent
    _supervisor_agent = None
