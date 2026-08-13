"""RunContext per-run 隔离测试 —— 多个并发请求互不污染。"""

from runtime.context import (
    RunContext,
    get_run_context,
    run_scope,
    run_context_to_dict,
    run_context_from_dict,
)


def test_two_contexts_are_isolated():
    a = RunContext(thread_id="thread-a")
    b = RunContext(thread_id="thread-b")

    with run_scope(a):
        get_run_context().record_cost(1.5)
        get_run_context().visited_urls.add("http://a.example")
        get_run_context().record_tool_call("researcher")

    with run_scope(b):
        # b 的 cost / visited_urls / 计数都是独立的，不受 a 影响
        assert get_run_context().cost == 0.0
        assert get_run_context().visited_urls == set()
        assert get_run_context().get_tool_call_count("researcher") == 0

    # a 的值保持不变
    assert a.cost == 1.5
    assert a.visited_urls == {"http://a.example"}
    assert a.get_tool_call_count("researcher") == 1


def test_scope_resets_after_exit():
    a = RunContext(thread_id="a")
    with run_scope(a):
        assert get_run_context() is a
    # 退出 run_scope 后回到默认上下文，不再是 a
    assert get_run_context() is not a


def test_evidence_dedup_by_source_id():
    ctx = RunContext(thread_id="t")
    with run_scope(ctx):
        ctx.add_evidence(url="https://x.example", quote="first")
        ctx.add_evidence(url="https://x.example", quote="second")
    assert len(ctx.evidence) == 1
    assert ctx.evidence[0].quote == "first"


def test_default_context_does_not_raise():
    """图外直接调用 get_run_context() 也能拿到兜底实例。"""
    ctx = get_run_context()
    assert ctx.thread_id == "__default__"


def test_record_cost_ignores_non_positive():
    ctx = RunContext(thread_id="t")
    with run_scope(ctx):
        get_run_context().record_cost(-1.0)
        get_run_context().record_cost(0.0)
    assert ctx.cost == 0.0


def test_run_context_snapshot_roundtrip():
    """resume 用的快照序列化/还原往返一致。"""
    a = RunContext(thread_id="t")
    a.cost = 1.23
    a.visited_urls = {"http://a", "http://b"}
    a.add_evidence(url="http://a", title="A", quote="q")
    a.record_tool_call("researcher")
    a.middleware_state["token"] = {"total": 10, "stopped": True}

    b = run_context_from_dict(run_context_to_dict(a))
    assert b.thread_id == "t"
    assert b.cost == 1.23
    assert b.visited_urls == {"http://a", "http://b"}
    assert len(b.evidence) == 1 and b.evidence[0].url == "http://a"
    assert b.get_tool_call_count("researcher") == 1
    assert b.middleware_state["token"]["stopped"] is True


def test_restore_from_mutates_in_place():
    """restore_from 原地覆盖，保证 run_scope 已绑定的实例仍有效。"""
    a = RunContext(thread_id="t", cost=0.0)
    b = RunContext(thread_id="t", cost=9.9)
    b.visited_urls.add("http://x")
    a.restore_from(run_context_to_dict(b))
    assert a.cost == 9.9
    assert a.visited_urls == {"http://x"}
