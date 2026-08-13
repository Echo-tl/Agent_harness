"""SummarizationMiddleware 异步压缩测试 —— abefore_model 走 ainvoke，不阻塞事件循环。"""

import pytest
from langchain_core.messages import HumanMessage

import middleware.summarization as summ_mod
from middleware.summarization import SummarizationMiddleware


class _FakeSummary:
    content = "压缩后的摘要"


class _FakeLLM:
    def __init__(self):
        self.ainvoke_calls = 0

    async def ainvoke(self, prompt):
        self.ainvoke_calls += 1
        return _FakeSummary()


@pytest.mark.asyncio
async def test_abefore_model_compresses_async(monkeypatch):
    fake = _FakeLLM()
    monkeypatch.setattr(summ_mod, "llm", fake)

    mid = SummarizationMiddleware(name="Test", threshold_chars=10)
    messages = [HumanMessage(content="x" * 500) for _ in range(10)]
    state = {"messages": messages}

    out = await mid.abefore_model(state, None)
    assert fake.ainvoke_calls == 1          # 走异步 ainvoke
    assert out is not None
    assert "研究摘要" in out["messages"][0].content   # 第一条是压缩摘要
    assert out["messages"][-1].content == messages[-1].content  # 最近消息保留


@pytest.mark.asyncio
async def test_abefore_model_skips_below_threshold(monkeypatch):
    fake = _FakeLLM()
    monkeypatch.setattr(summ_mod, "llm", fake)

    mid = SummarizationMiddleware(name="Test", threshold_chars=10000)
    state = {"messages": [HumanMessage(content="short")]}

    out = await mid.abefore_model(state, None)
    assert out is None
    assert fake.ainvoke_calls == 0


@pytest.mark.asyncio
async def test_sync_before_model_still_works(monkeypatch):
    class _SyncFake:
        def invoke(self, prompt):
            return _FakeSummary()

    monkeypatch.setattr(summ_mod, "llm", _SyncFake())

    mid = SummarizationMiddleware(name="Test", threshold_chars=10)
    state = {"messages": [HumanMessage(content="y" * 500) for _ in range(10)]}
    out = mid.before_model(state, None)
    assert out is not None
    assert "研究摘要" in out["messages"][0].content
