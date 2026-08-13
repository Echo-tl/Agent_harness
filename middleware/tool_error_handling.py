
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from runtime.context import get_run_context

class ToolErrorHandlingMiddleware(AgentMiddleware):
    """中间件：处理工具调用错误。工具异常被捕获并转为 ToolMessage，
    不崩整个 run —— 异常不传播即可视为"恢复"。

    埋点（telemetry.tool_error）：errors（捕获的工具异常数），
    供 benchmarks/ 评测"工具异常恢复率"。
    """

    key = "tool_error"

    @staticmethod
    def _record_error() -> None:
        get_run_context().tick("tool_error", "errors")

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except Exception as e:
            self._record_error()
            return ToolMessage(
                content=f"工具执行出错: {e}",
                tool_call_id=request.tool_call["id"],   # ←从 request 里取
            )

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as e:
            self._record_error()
            return ToolMessage(
                content=f"工具执行出错: {e}",
                tool_call_id=request.tool_call["id"],
            )
