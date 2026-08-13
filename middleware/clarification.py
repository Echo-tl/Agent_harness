
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

class ClarificationMiddleware(AgentMiddleware):

    key = "clarification"

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] != "ask_clarification":
            return handler(request)

        question = request.tool_call["args"]["question"]
        from langgraph.types import interrupt
        user_answer = interrupt(f"{question}")
        return ToolMessage(
            content=user_answer,
            tool_call_id=request.tool_call["id"],
            name="ask_clarification",
        )

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] != "ask_clarification":
            return await handler(request)

        question = request.tool_call["args"]["question"]
        from langgraph.types import interrupt
        user_answer = interrupt(f"{question}")
        return ToolMessage(
            content=user_answer,
            tool_call_id=request.tool_call["id"],
            name="ask_clarification",
        )