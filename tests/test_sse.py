"""SSE 事件统一结构测试 —— 所有事件都是 {type, message, data}，且平铺镜像兼容前端。"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from streaming import _event, _format_event
from runtime.context import RunContext, run_scope


def _parse_sse(sse: str) -> dict:
    return json.loads(sse[6:].strip())


@pytest.mark.asyncio
async def test_yield_stream_emits_progress_between_chunks():
    """工具执行中途 push 的进度会被实时发出（不用等整个 super-step 结束）。"""
    from streaming import _yield_stream

    ctx = RunContext("t")

    async def fake_stream():
        # astream chunk 格式是 (key, update)；父图节点 key=()、update={node: value}
        yield ((), {"planner": {"research_brief": "1. A"}})
        ctx.push_progress("正在爬取 1/2")
        ctx.push_progress("正在爬取 2/2")
        yield ((), {"final_report": {"final_report": "# 报告"}})

    events = []
    with run_scope(ctx):
        async for sse in _yield_stream(ctx, fake_stream(), 10):
            events.append(_parse_sse(sse))

    types = [e["type"] for e in events]
    assert "progress" in types
    assert "plan" in types
    assert "report" in types
    msgs = [e.get("message") for e in events if e["type"] == "progress"]
    assert "正在爬取 1/2" in msgs
    assert "正在爬取 2/2" in msgs


def test_event_unified_structure():
    ev = _event("plan", "研究计划已生成", research_brief="x")
    assert ev["type"] == "plan"
    assert ev["message"] == "研究计划已生成"
    assert ev["data"] == {"research_brief": "x"}
    # 平铺镜像：前端无需改动仍能读 event.research_brief
    assert ev["research_brief"] == "x"


def test_event_without_payload():
    ev = _event("complete", "研究已收集足够信息")
    assert ev["type"] == "complete"
    assert ev["message"] == "研究已收集足够信息"
    assert ev["data"] == {}


def test_format_event_plan():
    ev = _format_event((), "planner", {"research_brief": "1. 主题A"})
    assert ev["type"] == "plan"
    assert ev["data"]["research_brief"] == "1. 主题A"


def test_format_event_report_with_citation_stats():
    update = {
        "final_report": "# 报告",
        "citation_stats": {"coverage": 0.5, "groundedness": 1.0},
    }
    ev = _format_event((), "final_report", update)
    assert ev["type"] == "report"
    assert ev["content"] == "# 报告"
    assert ev["data"]["citation_stats"]["coverage"] == 0.5
    assert ev["citation_stats"]["groundedness"] == 1.0


def test_format_event_returns_none_for_internal_nodes():
    assert _format_event((), "some_internal_node", {"x": 1}) is None
    assert _format_event((), "planner", {}) is None


def _msg_with_tool_calls(*names):
    msg = AIMessage(content="")
    msg.tool_calls = [
        {"name": name, "args": {"research_topic": f"主题{i}"}, "id": f"c{i}"}
        for i, name in enumerate(names)
    ]
    return msg


def test_format_event_dispatch_on_task_tool_call():
    """当前图用 messages + task 工具：最后一条 AIMessage 调用 task → dispatch 事件。"""
    update = {"messages": [HumanMessage("hi"), _msg_with_tool_calls("task")]}
    ev = _format_event((), "supervisor", update)
    assert ev is not None
    assert ev["type"] == "dispatch"
    assert ev["data"]["topics"] == ["主题0"]


def test_format_event_complete_on_research_complete():
    update = {"messages": [_msg_with_tool_calls("ResearchComplete")]}
    ev = _format_event((), "supervisor", update)
    assert ev is not None
    assert ev["type"] == "complete"


def test_format_event_ignores_old_field_names():
    """旧的 supervisor_messages / ConductResearch 不再被读取。"""
    old_update = {
        "supervisor_messages": [_msg_with_tool_calls("ConductResearch")],
        "messages": [],
    }
    assert _format_event((), "supervisor", old_update) is None


def test_format_event_ignores_non_supervisor_tool_calls():
    """researcher 的 search 等工具调用不应触发 dispatch。"""
    update = {"messages": [_msg_with_tool_calls("search")]}
    assert _format_event((), "agent", update) is None
