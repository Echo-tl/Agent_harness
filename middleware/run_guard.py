"""RunGuardMiddleware —— 让迭代次数 / 成本上限真正生效。

`create_agent` 的 ReAct 循环默认会无限跑下去，直到模型自己停。
本中间件在每次模型返回后检查：

- 工具调用次数是否超过 `max_calls`（可按工具名过滤，如只统计 `task` 调用）
- 本次运行累计成本是否超过 `max_cost`

任一超限就把最后一条 AIMessage 的 tool_calls 清空，并注入提示要求
"基于现有结果总结"，从而强制 agent 收尾，而不是继续搜索。

计数存放在 RunContext（contextvar，按 thread_id 隔离）里 —— 不会像
之前用实例属性那样跨请求串线。
"""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from runtime.context import get_run_context


class RunGuardMiddleware(AgentMiddleware):
    """迭代 / 成本上限守卫。

    Args:
        counter_key: 在 RunContext.tool_call_counts 里使用的计数键，
                     不同的 agent 用不同的 key（如 "researcher"、"supervisor_task"）。
        max_calls: 工具调用次数上限；None 表示不限次数。
        tool_names: 只统计这些工具名的调用；None 表示统计全部工具。
        max_cost: 累计成本上限（美元）；None 表示不限成本。
    """

    key = "run_guard"

    def __init__(
        self,
        counter_key: str,
        max_calls: int | None = None,
        tool_names: list[str] | None = None,
        max_cost: float | None = None,
    ):
        super().__init__()
        self._counter_key = counter_key
        self._max_calls = max_calls
        self._tool_names = tool_names
        self._max_cost = max_cost

    def after_model(self, state, runtime):
        messages = state.get("messages", [])
        if not messages:
            return None
        last_msg = messages[-1]
        if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
            return None

        ctx = get_run_context()

        # 统计本轮命中的 tool_calls（按工具名过滤）
        relevant = [
            tc for tc in last_msg.tool_calls
            if self._tool_names is None or tc.get("name") in self._tool_names
        ]
        if not relevant:
            return None

        # ── 迭代上限 ──
        if self._max_calls is not None:
            count = ctx.get_tool_call_count(self._counter_key) + len(relevant)
            ctx.tool_call_counts[self._counter_key] = count
            if count > self._max_calls:
                ctx.tick(self.key, "calls_blocked")
                reason = f"已达到工具调用上限（{self._max_calls} 次）"
                return self._force_finish(messages, last_msg, reason)

        # ── 成本上限 ──
        if self._max_cost is not None and ctx.cost > self._max_cost:
            ctx.tick(self.key, "cost_blocks")
            reason = f"已达到成本上限（${self._max_cost:.4f}）"
            return self._force_finish(messages, last_msg, reason)

        return None

    @staticmethod
    def _force_finish(messages, last_msg: AIMessage, reason: str) -> dict:
        """清空 tool_calls，注入收尾提示。"""
        print(f"[RunGuard] {reason}，强制停止工具调用，基于现有结果总结")
        replacement = AIMessage(
            content=f"{last_msg.content}\n\n[系统提示：{reason}，请停止调用工具，直接基于现有结果给出总结。]",
            id=last_msg.id,
        )
        replacement.tool_calls = []
        return {"messages": [*messages[:-1], replacement]}
