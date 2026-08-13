"""RunGuardMiddleware 测试 —— 迭代 / 成本上限真正生效。"""

from langchain_core.messages import AIMessage, HumanMessage

from middleware.run_guard import RunGuardMiddleware
from runtime.context import RunContext, run_scope


def _msg_with_tool_calls(*names):
    msg = AIMessage(content="thinking")
    msg.tool_calls = [
        {"name": name, "args": {}, "id": f"call_{i}"}
        for i, name in enumerate(names)
    ]
    return msg


def test_guard_stops_after_max_calls():
    mid = RunGuardMiddleware(counter_key="researcher", max_calls=2)
    with run_scope(RunContext("t1")):
        # 前 2 次搜索：未超限，不干预
        for _ in range(2):
            state = {"messages": [HumanMessage("hi"), _msg_with_tool_calls("search")]}
            assert mid.after_model(state, None) is None

        # 第 3 次：超限，清空 tool_calls
        state = {"messages": [HumanMessage("hi"), _msg_with_tool_calls("search")]}
        out = mid.after_model(state, None)
        assert out is not None
        assert out["messages"][-1].tool_calls == []
        assert "上限" in out["messages"][-1].content


def test_guard_filters_by_tool_name():
    mid = RunGuardMiddleware(counter_key="supervisor_task", tool_names=["task"], max_calls=1)
    with run_scope(RunContext("t2")):
        # ResearchComplete 不计入 task 上限
        state0 = {"messages": [HumanMessage("hi"), _msg_with_tool_calls("ResearchComplete")]}
        assert mid.after_model(state0, None) is None

        # 1 次 task：未超
        state1 = {"messages": [HumanMessage("hi"), _msg_with_tool_calls("task")]}
        assert mid.after_model(state1, None) is None

        # 第 2 次 task：超限
        state2 = {"messages": [HumanMessage("hi"), _msg_with_tool_calls("task")]}
        assert mid.after_model(state2, None) is not None


def test_guard_enforces_cost_limit():
    mid = RunGuardMiddleware(counter_key="r", max_cost=1.0)
    with run_scope(RunContext("t3")) as ctx:
        state0 = {"messages": [HumanMessage("hi"), _msg_with_tool_calls("search")]}
        assert mid.after_model(state0, None) is None

        ctx.record_cost(2.0)  # 成本超限
        state1 = {"messages": [HumanMessage("hi"), _msg_with_tool_calls("search")]}
        out = mid.after_model(state1, None)
        assert out is not None
        assert out["messages"][-1].tool_calls == []


def test_guard_does_not_interfere_without_tool_calls():
    mid = RunGuardMiddleware(counter_key="r", max_calls=0, max_cost=0.0)
    with run_scope(RunContext("t4")) as ctx:
        ctx.record_cost(5.0)  # 已超成本
        state = {"messages": [HumanMessage("hi"), AIMessage(content="final answer")]}
        # 最后一条没有 tool_calls，守卫不干预
        assert mid.after_model(state, None) is None
