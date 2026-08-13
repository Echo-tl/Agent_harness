"""循环检测 —— 检测到重复的 tool_call 后强制停止，防止死循环。

历史记录按 name 存在 RunContext 里（每次请求独立），
不再用实例属性 —— 避免多个并发请求共用一个 agent 实例时互相污染。

埋点（telemetry.loop_detection）：repetition_events / forced_stops / max_depth，
供 benchmarks/ 评测"循环抑制率"。
"""

from langchain.agents.middleware import AgentMiddleware

from runtime.context import get_run_context

class LoopDetectionMiddleware(AgentMiddleware):
    """对最近 N 轮 tool_call 做 hash，同一调用出现 >=5 次就清空 tool_calls"""

    key = "loop_detection"

    def __init__(self, name: str = "loop_detection"):
        super().__init__()
        self._name = name

    def after_model(self, state, runtime):
        ctx = get_run_context()
        st = ctx.middleware_state.setdefault(self._name, {"history": []})
        history = st["history"]

        # 1. 取最后一条 AIMessage 的 tool_calls
        messages = state.get("messages", [])
        if not messages:
            return None
        last_message = messages[-1]

        # 2. 算 hash(tool_call["name"] + tool_call["args"])
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_hash = hash(tool_call["name"] + str(tool_call["args"]))
                if tool_hash in history:
                    ctx.tick(self.key, "repetition_events")
                history.append(tool_hash)
                if len(history) > 20:  # 只保留最近 20 个
                    del history[:-20]

                # 3. 查这个 hash 出现了几次
                count = history.count(tool_hash)
                # 4. 记录最深重复深度（评测用）
                t = ctx.telemetry.setdefault(self.key, {})
                if count > t.get("max_depth", 0):
                    t["max_depth"] = count

                # 5. >= 5 次 →清空 tool_calls，强制 LLM 给文字回复
                if count >= 5:
                    ctx.tick(self.key, "forced_stops")
                    print(f"[LoopDetectionMiddleware] 检测到循环调用 {tool_call['name']}，强制清空 tool_calls")
                    return {
                        "messages": [
                            *messages[:-1],  # 保留之前的消息
                            last_message.copy(update={"tool_calls": []})  # 清空 tool_calls
                        ]
                    }
        return None
