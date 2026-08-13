"""Compress 节点 —— 将原始搜索结果压缩为结构化研究笔记"""

from state import ResearcherState, ResearcherOutputState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from state.models import CompressedResearch
from prompts import COMPRESS_PROMPT


# 绑定结构化输出
from llm_config import llm as compress_llm
from llm_config import add_cost, estimate_llm_cost
structured_llm = compress_llm.with_structured_output(CompressedResearch, method="function_calling")

async def compress_node(state: ResearcherState) -> dict:
    print(f"\n[Compress] compressing search results...")

    # 提取所有搜索结果（和 answer 节点一样，构建干净上下文）
    search_results = []
    for msg in state["messages"]:
        print(f"  [{type(msg).__name__}] content_len={len(str(msg.content))}")
        if isinstance(msg, ToolMessage):
            search_results.append(msg.content)

    results_text = "\n\n---\n\n".join(search_results) if search_results else "无搜索结果"

    research_topic = state["research_topic"]

    # 调用结构化 LLM（异步，避免阻塞事件循环）
    print(f"[Compress] sending to LLM, results length: {len(results_text)} chars...")
    try:
        response = await structured_llm.ainvoke([
            SystemMessage(content=COMPRESS_PROMPT),
            HumanMessage(content=f"用户问题：{research_topic}\n\n搜索结果：\n{results_text}"),
        ])

        print(f"[Compress] LLM response received, type={type(response).__name__}")
    except Exception as e:
        print(f"[Compress] LLM error: {e}")
        return {
            "compressed_research": results_text[:2000],
            "raw_notes": [results_text[:2000]],
        }

    # structured output 可能返回 None（模型不支持 function calling 时）
    if response is None:
        print(f"[Compress] structured output returned None, using raw text instead")
        return {
            "compressed_research": results_text[:3000],
            "raw_notes": [results_text[:3000]],
        }

    # response 是 CompressedResearch 对象（每条 note 自带 evidences）
    notes_text = f"## 总体概括\n{response.summary}\n\n"
    notes_text += "## 研究笔记\n"

    for i, note in enumerate(response.notes, 1):
        notes_text += f"\n### 笔记 {i}: {note.topic}\n"
        notes_text += f"**核心发现**: {note.key_finding}\n"
        notes_text += f"**详细说明**: {note.details}\n"
        # 把每条证据的来源 URL 一并写进笔记文本，报告阶段可直接引用
        if note.evidences:
            notes_text += "**来源**: " + ", ".join(
                f"[{ev.title or ev.url}]({ev.url})" for ev in note.evidences
            ) + "\n"

    add_cost(estimate_llm_cost(
        input_chars=len(results_text),
        output_chars=len(notes_text),
    ))

    print(f"[Compress] done: {len(response.notes)} notes")

    return {
        "compressed_research": notes_text,
        "raw_notes": [notes_text],
    }
