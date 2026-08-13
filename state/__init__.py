"""
State 定义 —— Agent 的"数据契约"

为什么 State 是 Agent 设计的起点？
━━━━━━━━━━━━━━━━━━━━━━━━━━
1. State 决定了"什么信息在节点之间流动"
2. 每个 Node 的输入/输出都是 State 的子集
3. LangGraph 的 StateGraph 以 State 为核心：定义好 State，Graph 的结构就自然浮现

设计原则（本次 V1）：
- 每个字段对应一个流程阶段产生的数据
- 使用 TypedDict 提供类型安全（IDE 自动补全 + 类型检查）
- 字段名要语义化，让人一眼看出它的含义

V1 的 State 设计思路：
  question       → 用户输入，第一个节点需要
  search_results → 搜索节点产出，总结节点消费
  summary        → 总结节点产出，回答节点消费
  final_answer   → 回答节点产出，最终展示给用户
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class MultiAgentState(TypedDict):
    """多 Agent 协作的全局状态

    设计思路：
    - messages: 记录所有 Agent 的对话历史，供后续分析和调试
    - 每个 Agent 的输入输出都可以被记录在 messages 中，形成完整的"事件日志"
    - 这样不仅有助于调试，还可以为未来的"Agent 记忆"功能打基础
    """
    messages: Annotated[list[BaseMessage], add_messages] #新增对话历史
    total_cost: float                                   # 累计花费
    research_brief: str # 研究简介，供 Agent 参考
    final_report: str # 最终报告，供展示节点使用
    citation_stats: dict  # 引用校验统计（coverage / groundedness / 缺失引用），供前端展示
    run_ctx_data: dict  # RunContext 快照：请求被 interrupt 时持久化，resume 时完整恢复


class ResearcherState(TypedDict):
    """研究 Agent 的全局状态"""
    messages: Annotated[list[BaseMessage], add_messages] #新增对话历史
    research_topic: str # 研究主题，供 Agent 参考

class ResearcherOutputState(TypedDict):
    """研究 Agent 的输出状态"""
    compressed_research: str # 最终回答，供展示节点使用
    raw_notes: list[str] # 原始笔记列表，供压缩节点使用
