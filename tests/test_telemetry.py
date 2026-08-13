"""中间件 telemetry 埋点测试 —— 评测指标的计数来源。"""

from langchain_core.messages import AIMessage, HumanMessage

from middleware.loop_detection import LoopDetectionMiddleware
from middleware.token_budget import TokenBudgetMiddleware
from middleware.tool_error_handling import ToolErrorHandlingMiddleware
from middleware.run_guard import RunGuardMiddleware
from runtime.context import RunContext, run_scope


def _tool_msg(name="search", args=None):
    m = AIMessage(content="")
    m.tool_calls = [{"name": name, "args": args or {"query": "x"}, "id": "c1"}]
    return m


def test_loop_detection_telemetry():
    mid = LoopDetectionMiddleware(name="loop_t")
    ctx = RunContext("t1")
    with run_scope(ctx):
        for _ in range(5):
            mid.after_model({"messages": [HumanMessage("hi"), _tool_msg()]}, None)
    t = ctx.telemetry["loop_detection"]
    assert t["forced_stops"] == 1        # 第 5 次重复时强制停止
    assert t["repetition_events"] == 4   # 第 2~5 次出现都属于重复
    assert t["max_depth"] == 5


def test_token_budget_telemetry():
    mid = TokenBudgetMiddleware(name="token_t", max_tokens=100)
    ctx = RunContext("t2")
    with run_scope(ctx):
        out = mid.after_model(
            {"messages": [HumanMessage(content="x" * 500), _tool_msg()]}, None
        )
    assert out is not None  # 超预算 → 强制停止
    t = ctx.telemetry["token_budget"]
    assert t["forced_stops"] == 1
    assert t["max_chars"] == 100
    assert t["total_chars"] > 100


def test_tool_error_telemetry():
    class Req:
        tool_call = {"id": "c1"}

    def handler(req):
        raise RuntimeError("boom")

    mid = ToolErrorHandlingMiddleware()
    ctx = RunContext("t3")
    with run_scope(ctx):
        result = mid.wrap_tool_call(Req(), handler)
    assert "工具执行出错" in result.content   # 异常被恢复为 ToolMessage
    assert ctx.telemetry["tool_error"]["errors"] == 1


def test_run_guard_telemetry():
    mid = RunGuardMiddleware(counter_key="r", max_calls=1)
    ctx = RunContext("t4")
    state = {"messages": [HumanMessage("hi"), _tool_msg()]}
    with run_scope(ctx):
        assert mid.after_model(state, None) is None   # 第 1 次未超限
        assert mid.after_model(state, None) is not None  # 第 2 次超限
    assert ctx.telemetry["run_guard"]["calls_blocked"] == 1
