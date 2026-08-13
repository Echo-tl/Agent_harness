from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from runtime.context import get_run_context

class TokenBudgetMiddleware(AgentMiddleware):
    """token 预算超限后强制停止 tool_calls，防止死循环。

    状态按 name 存在 RunContext 里（每次请求独立），
    不再用实例属性 —— 避免多个并发请求共用一个 agent 实例时互相污染。

    埋点（telemetry.token_budget）：total_chars / max_chars / forced_stops，
    供 benchmarks/ 评测"Token 消耗降低 / 超预算拦截"。
    """

    key = "token_budget"

    def __init__(self, name: str = "token_budget", max_tokens: int = 30000):
        super().__init__()
        self._name = name
        self._max = max_tokens

    def after_model(self, state, runtime):
        ctx = get_run_context()
        st = ctx.middleware_state.setdefault(self._name, {"total": 0, "stopped": False})
        if st["stopped"]:
            return None

        messages = state.get("messages", [])
        st["total"] = sum(len(str(m.content)) for m in messages if hasattr(m, "content")) // 2

        # 埋点：记录本轮观测的 token 估算与预算上限
        ctx.telemetry.setdefault(self.key, {}).update(
            {"total_chars": st["total"], "max_chars": self._max}
        )

        if st["total"] > self._max:
            st["stopped"] = True
            last_msg = messages[-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                print(f"[TokenBudget] {st['total']}/{self._max} tokens, 强制停止工具调用")
                ctx.tick(self.key, "forced_stops")
                # 新建一个 AIMessage 替换（确保 tool_calls 为空）
                replacement = AIMessage(
                    content=last_msg.content,
                    id=last_msg.id,
                )
                replacement.tool_calls = []
                return {"messages": [replacement]}

        return None
