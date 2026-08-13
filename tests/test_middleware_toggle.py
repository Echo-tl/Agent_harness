"""中间件开关（A/B 基线）测试 —— enabled_middleware 过滤与图构建。"""

import config as config_mod
from middleware import enabled_middleware
from middleware.summarization import SummarizationMiddleware
from middleware.run_guard import RunGuardMiddleware


def _mws():
    return [SummarizationMiddleware("a"), RunGuardMiddleware(counter_key="r")]


def test_enabled_middleware_filters(monkeypatch):
    monkeypatch.setitem(config_mod.CONFIG["middleware"], "enabled", ["run_guard"])
    out = enabled_middleware(_mws())
    assert len(out) == 1
    assert isinstance(out[0], RunGuardMiddleware)


def test_enabled_middleware_empty_list_loads_all(monkeypatch):
    monkeypatch.setitem(config_mod.CONFIG["middleware"], "enabled", [])
    assert len(enabled_middleware(_mws())) == 2


def test_enabled_middleware_unknown_key_dropped(monkeypatch):
    monkeypatch.setitem(config_mod.CONFIG["middleware"], "enabled", ["nope"])
    assert enabled_middleware(_mws()) == []


def test_graph_builds_with_reduced_middleware(monkeypatch):
    """基线配置（只留 run_guard）下父图仍能构建。"""
    monkeypatch.setitem(config_mod.CONFIG["middleware"], "enabled", ["run_guard"])
    from nodes.supervisor import reset_supervisor_agent
    reset_supervisor_agent()
    try:
        from graph import build_parent_graph
        from langgraph.checkpoint.memory import InMemorySaver
        g = build_parent_graph(checkpointer=InMemorySaver())
        assert g is not None
    finally:
        reset_supervisor_agent()  # 复原缓存，避免影响其它测试
