"""利用率驱动的渐进式压缩测试 —— SummarizationMiddleware 内实现。

覆盖：
- 利用率落在 [50%, 85%)：轻量级截断旧工具结果（不调 LLM）。
- 利用率 ≥ 85%：auto-compact（LLM 全量摘要）。
- 利用率写入 RunContext.utilization。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import middleware.summarization as summ_mod
from middleware.summarization import SummarizationMiddleware
from runtime.context import RunContext, run_scope


class _FakeSummary:
    content = "压缩后的摘要"


class _FakeLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return _FakeSummary()

    async def ainvoke(self, prompt):
        self.calls += 1
        return _FakeSummary()


@pytest.mark.asyncio
async def test_auto_compact_on_high_utilization(monkeypatch):
    """利用率 ≥0.85 → auto-compact（即使没超绝对阈值）。"""
    fake = _FakeLLM()
    monkeypatch.setattr(summ_mod, "llm", fake)
    # 绝对阈值设极大（不触发），预算极小 → 利用率 ≥0.85
    mid = SummarizationMiddleware(
        name="T", threshold_chars=10**9, max_tokens=100
    )
    messages = [HumanMessage(content="x" * 500) for _ in range(10)]  # est≈2500, u≈25
    ctx = RunContext("t1")
    with run_scope(ctx):
        out = await mid.abefore_model({"messages": messages}, None)
    assert out is not None
    t = ctx.telemetry["summarization"]
    assert t["auto_compacts"] == 1
    assert t["compressions"] == 1
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_tighten_in_mid_band(monkeypatch):
    """利用率 ∈ [50%, 85%) → 轻量级截断旧 ToolMessage，不调 LLM。"""
    fake = _FakeLLM()
    monkeypatch.setattr(summ_mod, "llm", fake)
    mid = SummarizationMiddleware(
        name="T", threshold_chars=10**9, max_tokens=10000
    )
    # total chars = 6000 + 3000*3 = 15000 → est 7500 → u=0.75 ∈ [0.5, 0.85)
    big = ToolMessage(content="z" * 6000, tool_call_id="c1")
    messages = [big, HumanMessage("a" * 3000), HumanMessage("b" * 3000), HumanMessage("c" * 3000)]
    ctx = RunContext("t2")
    with run_scope(ctx):
        out = await mid.abefore_model({"messages": messages}, None)
    assert out is not None
    # 旧的 ToolMessage（index 0）被截短
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert len(tool_msgs[0].content) <= 800 + 20
    t = ctx.telemetry["summarization"]
    assert t["tightenings"] == 1
    assert t.get("auto_compacts", 0) == 0
    assert fake.calls == 0  # 轻量截断不调 LLM


def test_no_action_below_tighten_band():
    """利用率 < 50% → 不动，仅写入 utilization。"""
    mid = SummarizationMiddleware(
        name="T", threshold_chars=10**9, max_tokens=10000
    )
    ctx = RunContext("t3")
    with run_scope(ctx):
        out = mid.before_model({"messages": [HumanMessage(content="x" * 2000)]}, None)
    assert out is None                       # est=1000, u=0.1 < 0.5
    assert ctx.utilization == pytest.approx(0.1)


def test_utilization_computed_with_real_budget():
    """利用率 = 预估 token / 预算，写进 RunContext。"""
    mid = SummarizationMiddleware(
        name="T", threshold_chars=10**9, max_tokens=10000
    )
    ctx = RunContext("t4")
    with run_scope(ctx):
        mid.before_model({"messages": [HumanMessage(content="x" * 10000)]}, None)
    # est = 5000, u = 0.5（处于收紧带，但消息里没有可截断的 ToolMessage → 返回 None）
    assert ctx.utilization == pytest.approx(0.5)
