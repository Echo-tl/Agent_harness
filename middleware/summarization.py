"""上下文压缩 —— 利用率驱动的渐进式压缩（全部实现在本中间件内）。

三层递进（从轻量截断到全量摘要）：
1. 利用率 < tighten_utilization（默认 50%）：不动。
2. 利用率 ∈ [50%, 85%)：**轻量级截断** —— 把较旧的工具结果(ToolMessage)
   内容截短，收紧上下文体积，不调 LLM。
3. 利用率 ≥ auto_compact_utilization（默认 85%）或预估 token 超绝对阈值：
   **auto-compact** —— 对历史消息做"滑动窗口(保留最近 40%) + LLM 全量摘要"。

利用率 = 预估 token（字符数/2）/ 上下文预算（max_tokens）。每次 model 调用前把
当前利用率写进 RunContext.utilization，供评测与下游读取。

同时实现同步 before_model 与异步 abefore_model：agent 用 ainvoke/astream 时走异步版。

埋点（telemetry.summarization）：compressions / auto_compacts / tightenings /
chars_before / chars_after / utilization。
"""

from langchain.agents.middleware import AgentMiddleware
from llm_config import llm
from langchain_core.messages import HumanMessage, ToolMessage

from runtime.context import get_run_context

_MAX_TOOL_CHARS = 800  # 轻量截断时单条工具结果保留的字符上限


class SummarizationMiddleware(AgentMiddleware):
    key = "summarization"

    def __init__(
        self,
        name: str = "?",
        threshold_chars: int | None = None,
        max_tokens: int | None = None,
        tighten_utilization: float | None = None,
        auto_compact_utilization: float | None = None,
    ):
        super().__init__()
        self._name = name
        from config import CONFIG
        sc = CONFIG.get("summarization", {})
        self._threshold_chars = (
            threshold_chars if threshold_chars is not None else sc.get("threshold_chars", 3000)
        )
        self._max_tokens = max_tokens if max_tokens is not None else sc.get("max_tokens", 10000)
        self._tighten_u = (
            tighten_utilization if tighten_utilization is not None
            else sc.get("tighten_utilization", 0.50)
        )
        self._auto_u = (
            auto_compact_utilization if auto_compact_utilization is not None
            else sc.get("auto_compact_utilization", 0.85)
        )

    # ── 决策 ──
    def _compute_utilization(self, messages, ctx) -> tuple[int, float]:
        total_chars = sum(len(m.content) for m in messages if hasattr(m, "content"))
        est_tokens = total_chars // 2
        u = est_tokens / self._max_tokens if self._max_tokens else 0.0
        ctx.utilization = u
        ctx.telemetry.setdefault(self.key, {}).update({"utilization": round(u, 3)})
        return est_tokens, u

    def _should_compact(self, est_tokens: int, u: float) -> bool:
        return est_tokens > self._threshold_chars or u >= self._auto_u

    # ── L2 轻量级截断（50%~85%，不调 LLM）──
    @classmethod
    def _truncate_old_tool_messages(cls, messages, keep_ratio: float = 0.5) -> list:
        """把较旧的 ToolMessage 内容截短，收紧工具结果体积。"""
        split = int(len(messages) * keep_ratio)
        new = []
        for i, m in enumerate(messages):
            if (
                i < split
                and isinstance(m, ToolMessage)
                and isinstance(m.content, str)
                and len(m.content) > _MAX_TOOL_CHARS
            ):
                new.append(
                    m.model_copy(update={"content": m.content[:_MAX_TOOL_CHARS] + "\n...(已截断)"})
                )
            else:
                new.append(m)
        return new

    def _tighten(self, messages, ctx) -> dict | None:
        before = sum(len(str(m.content)) for m in messages if hasattr(m, "content"))
        new = self._truncate_old_tool_messages(messages)
        after = sum(len(str(m.content)) for m in new if hasattr(m, "content"))
        if before == after:
            return None  # 没有可收紧的旧工具结果
        print(f"[Summarization:{self._name}] utilization={ctx.utilization:.2f}，轻量截断旧工具结果 {before}→{after} 字符")
        ctx.tick(self.key, "tightenings")
        t = ctx.telemetry.setdefault(self.key, {})
        t["chars_before_tighten"] = t.get("chars_before_tighten", 0) + before
        t["chars_after_tighten"] = t.get("chars_after_tighten", 0) + after
        return {"messages": new}

    # ── L3 auto-compact（≥85% 或超绝对阈值，LLM 全量摘要）──
    @staticmethod
    def _split(messages):
        """前 60% 压缩，后 40% 保留（滑动窗口）。"""
        split = int(len(messages) * 0.6)
        old = messages[:split]
        recent = messages[split:]
        old_text = "\n".join(m.content for m in old)
        return old, recent, old_text

    def _record(self, messages_before, result_messages, auto: bool) -> None:
        before = sum(len(m.content) for m in messages_before if hasattr(m, "content"))
        after = sum(len(m.content) for m in result_messages if hasattr(m, "content"))
        ctx = get_run_context()
        ctx.tick(self.key, "compressions")
        if auto:
            ctx.tick(self.key, "auto_compacts")
        t = ctx.telemetry.setdefault(self.key, {})
        t["chars_before"] = t.get("chars_before", 0) + before
        t["chars_after"] = t.get("chars_after", 0) + after

    @staticmethod
    def _result(recent, summary) -> dict:
        return {
            "messages": [
                HumanMessage(f"<研究摘要>\n{summary.content}\n</研究摘要>"),
                *recent,
            ]
        }

    # ── 入口 ──
    def before_model(self, state, runtime):
        """同步版：agent 用 invoke/stream（同步）时调用。"""
        messages = state.get("messages", [])
        ctx = get_run_context()
        est, u = self._compute_utilization(messages, ctx)

        if self._should_compact(est, u):
            print(f"[Summarization:{self._name}] auto-compact（u={u:.2f}, est={est}）")
            _, recent, old_text = self._split(messages)
            summary = llm.invoke(f"请将以下内容压缩成简短摘要，保留关键信息：\n{old_text}")
            result = self._result(recent, summary)
            self._record(messages, result["messages"], auto=(u >= self._auto_u))
            return result

        if u >= self._tighten_u:
            return self._tighten(messages, ctx)

        return None

    async def abefore_model(self, state, runtime):
        """异步版：agent 用 ainvoke / astream 时调用，不阻塞事件循环。"""
        messages = state.get("messages", [])
        ctx = get_run_context()
        est, u = self._compute_utilization(messages, ctx)

        if self._should_compact(est, u):
            print(f"[Summarization:{self._name}] auto-compact（u={u:.2f}, est={est}）")
            _, recent, old_text = self._split(messages)
            summary = await llm.ainvoke(f"请将以下内容压缩成简短摘要，保留关键信息：\n{old_text}")
            result = self._result(recent, summary)
            self._record(messages, result["messages"], auto=(u >= self._auto_u))
            return result

        if u >= self._tighten_u:
            return self._tighten(messages, ctx)

        return None
