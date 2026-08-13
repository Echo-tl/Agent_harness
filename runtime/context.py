"""RunContext —— per-run（按 thread_id）状态，通过 contextvar 在请求内传递。

问题背景
────────
之前 `_visited_urls`（tools/__init__.py）与 `_cost_tracker`（llm_config.py）是
模块级全局变量，每次请求开头调用 reset，多个用户并发请求时互相清空 / 覆盖。

本模块的解法
────────
每次请求（thread_id）在启动时创建一个 `RunContext`，用 `run_scope` 注入到
当前 asyncio task 的 contextvar。图的节点、工具、middleware 都在同一个 task
内执行，contextvar 自动传播，因此它们拿到的都是"本次运行"的独立状态。
请求结束后 reset，天然按 thread_id 隔离 —— 不再使用模块级可变全局变量。

注意
────
- 同步工具（rag_search / memory_search）在线程池执行时，LangGraph 会拷贝
  当前 context 到线程，因此 contextvar 依然可见。
- 在图外直接调用工具（如测试）时，`get_run_context()` 返回进程级兜底实例，
  保证不抛异常。
"""

from __future__ import annotations

import contextvars
import copy
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from state.models import Evidence

# 当前运行上下文：每个 asyncio task 一个，值为 None 表示未在运行中
_current: "contextvars.ContextVar[Optional[RunContext]]" = contextvars.ContextVar(
    "mini_deerflow_run_context", default=None
)

# 兜底实例（图外调用工具时用）。用锁保证单线程初始化。
_default: Optional["RunContext"] = None
_default_lock = threading.Lock()


@dataclass
class RunContext:
    """一次研究运行的独立状态。

    每个 thread_id 独立一份，存放 visited URLs、累计成本、证据列表、
    工具调用计数等，避免模块级全局变量的并发污染。
    """

    thread_id: str
    visited_urls: set[str] = field(default_factory=set)
    cost: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    # 按 agent/用途区分的工具调用计数（如 "researcher"、"supervisor_task"）
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    # middleware 的 per-run 临时状态（token budget、循环检测等），按 name 隔离
    middleware_state: dict[str, dict] = field(default_factory=dict)
    # 中间件埋点计数（评测用）：key → {counters}
    telemetry: dict[str, dict] = field(default_factory=dict)
    # 当前上下文利用率（预估 token / 预算），由 SummarizationMiddleware 维护
    utilization: float = 0.0
    # 进度消息（节点/工具 push，streaming 每步 drain 后作为 SSE progress 事件发送）
    progress_events: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)

    # ── 成本 ──
    def record_cost(self, amount: float) -> None:
        """累加一次 LLM 调用的估算花费（美元）"""
        if amount and amount > 0:
            self.cost += amount

    # ── 证据 ──
    def add_evidence(
        self,
        url: str,
        title: str = "",
        quote: str = "",
        published_at: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> Evidence:
        """记录一条来源证据（去重，按 URL 判断）"""
        sid = source_id or url
        for ev in self.evidence:
            if ev.source_id == sid:
                return ev
        ev = Evidence(
            source_id=sid,
            url=url,
            title=title,
            quote=quote[:500],
            published_at=published_at,
        )
        self.evidence.append(ev)
        return ev

    # ── 工具调用计数 ──
    def record_tool_call(self, key: str = "default") -> None:
        self.tool_call_counts[key] = self.tool_call_counts.get(key, 0) + 1

    def get_tool_call_count(self, key: str) -> int:
        return self.tool_call_counts.get(key, 0)

    def snapshot(self) -> dict:
        """供报告/SSE 展示的结构化摘要"""
        return {
            "thread_id": self.thread_id,
            "cost": round(self.cost, 6),
            "visited_urls": len(self.visited_urls),
            "evidence_count": len(self.evidence),
            "tool_call_count": sum(self.tool_call_counts.values()),
        }

    def restore_from(self, data: Optional[dict]) -> None:
        """从快照 dict 原地恢复状态。

        原地（in-place）覆盖而非替换对象，保证 run_scope 已绑定的实例
        依然有效（resume 请求里 ContextVar 指向的正是这个实例）。
        """
        data = data or {}
        self.thread_id = data.get("thread_id", self.thread_id)
        self.cost = float(data.get("cost", self.cost))
        self.visited_urls = set(data.get("visited_urls", []))
        self.evidence = [Evidence(**ev) for ev in data.get("evidence", [])]
        self.tool_call_counts = dict(data.get("tool_call_counts", {}))
        self.middleware_state = copy.deepcopy(data.get("middleware_state", {}))
        self.telemetry = copy.deepcopy(data.get("telemetry", {}))
        self.utilization = float(data.get("utilization", 0.0))

    def tick(self, key: str, field: str, delta: int = 1) -> None:
        """中间件埋点：累加 telemetry[key][field] += delta。"""
        bucket = self.telemetry.setdefault(key, {})
        bucket[field] = bucket.get(field, 0) + delta

    def push_progress(self, message: str) -> None:
        """记录一条进度消息，streaming 在下一个 chunk 后作为 SSE progress 事件发送。"""
        if message:
            self.progress_events.append(message)


def get_run_context() -> RunContext:
    """返回当前运行上下文。

    在 `run_scope` 内返回本次请求的 RunContext；否则返回进程级兜底实例
    （保证工具脱离图运行时也可用，且不抛异常）。
    """
    ctx = _current.get()
    if ctx is not None:
        return ctx
    return _get_default()


def _get_default() -> RunContext:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = RunContext(thread_id="__default__")
    return _default


class run_scope:
    """把 RunContext 注入当前上下文，块结束后恢复。

    用法：:

        ctx = RunContext(thread_id="t1")
        with run_scope(ctx):
            ...  # 期间 get_run_context() 返回 ctx
        # 结束后恢复为进入前的上下文
    """

    def __init__(self, ctx: RunContext):
        self._ctx = ctx
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> RunContext:
        self._token = _current.set(self._ctx)
        return self._ctx

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None
        return False


def run_context_to_dict(ctx: RunContext) -> dict:
    """把 RunContext 序列化成可入 checkpoint 的纯 dict（JSON 安全）。

    用于流式 resume：请求被 interrupt 时把快照写进图状态（run_ctx_data），
    下一次 resume 请求再把它完整还原，保证同一对话跨请求共享状态。
    """
    return {
        "thread_id": ctx.thread_id,
        "cost": ctx.cost,
        "visited_urls": sorted(ctx.visited_urls),
        "evidence": [ev.model_dump() for ev in ctx.evidence],
        "tool_call_counts": dict(ctx.tool_call_counts),
        "middleware_state": copy.deepcopy(ctx.middleware_state),
        "telemetry": copy.deepcopy(ctx.telemetry),
        "utilization": ctx.utilization,
    }


def run_context_from_dict(data: Optional[dict]) -> RunContext:
    """从快照 dict 还原 RunContext（供 resume 请求完整恢复）。"""
    data = data or {}
    ctx = RunContext(thread_id=data.get("thread_id", "restored"))
    ctx.cost = float(data.get("cost", 0.0))
    ctx.visited_urls = set(data.get("visited_urls", []))
    ctx.evidence = [Evidence(**ev) for ev in data.get("evidence", [])]
    ctx.tool_call_counts = dict(data.get("tool_call_counts", {}))
    ctx.middleware_state = copy.deepcopy(data.get("middleware_state", {}))
    return ctx
