""" Planner 节点：根据用户问题生成研究计划
"""
from langchain_core.messages import SystemMessage, HumanMessage
from state import MultiAgentState
from prompts import PLANNER_PROMPT, SUPERVISOR_SYSTEM_PROMPT

async def planner_node(state: MultiAgentState) -> dict:
    """Planner 节点：根据用户问题生成研究计划

    输入：state["messages"]
    输出：{"plan": "1. ... 2. ... 3. ..."}

    这个节点的作用是将用户的模糊问题拆解成具体的研究步骤，指导后续的搜索和总结。
    异步实现，用 ainvoke 避免阻塞 FastAPI 事件循环。
    """
    # 取用户的第一条消息作为原始问题
    question = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) and msg.content.strip():
            question = msg.content
            break
    if not question:
        question = state["messages"][-1].content  # 兜底

    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=question),
    ]

    from llm_config import llm as planner_llm
    response = await planner_llm.ainvoke(messages)

    # 解析LLM返回的文本为列表
    # LLM 返回格式: "1. 子问题A\n2. 子问题B\n3. 子问题C"
    raw_plan = response.content
    plan_items = []
    for line in raw_plan.strip().split("\n"):
        line = line.strip()
         # 去掉前面的序号 "1. " "2. " 等
        if line and (line[0].isdigit() or line.startswith("-")):
            for sep in [".", "-", "、"]:
                if sep in line:
                    line = line.split(sep, 1)[1]
                    break
            if line.strip():
                plan_items.append(line.strip())
    
    supervisor_messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=raw_plan), # 把整个 plan 作为第一条 Human 消息
    ]

    return {
        "research_brief": "\n".join(plan_items),
        "messages": [
          SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
          HumanMessage(content=f"研究主题: {question}\n\n研究计划:\n{raw_plan}\n\n请按计划逐一调用 task 工具进行研究。"),
        ]
    }