
from datetime import datetime
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage

class DynamicContextMiddleware(AgentMiddleware):
    """注入当前日期到第一条HUmanmessage中，方便 LLM 生成与时间相关的回答"""

    key = "dynamic_context"

    def before_agent(self, state, runtime):
        messages = state.get("messages", [])
            
        if not messages:
            return None
        
        # 1. 检查是否已经注入过
        for msg in messages:
            content = msg.content if hasattr(msg, "content") else ""
            if isinstance(content, str) and "<system-reminder>" in content:
                return None  # 已经注入过了
         
        # 2. 没找到 →第一次，注入
        current_date = datetime.now().strftime("%Y-%m-%d, %A")
        reminder = f"<system-reminder>\n当前日期: {current_date}\n</system-reminder>"

        return {
            "messages": [
                HumanMessage(content=reminder),
                *messages
            ]
        }


