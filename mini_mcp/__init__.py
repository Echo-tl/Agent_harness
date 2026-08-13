"""MCP (Model Context Protocol) 集成 ——接入外部工具和数据源"""

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from .client import MCPClientManager
from .tool_selector import MCPToolSelector
from .research import MCPResearchSkill

__all__  = ["MCPClientManager", "MCPToolSelector", "MCPResearchSkill", "HAS_MCP"]