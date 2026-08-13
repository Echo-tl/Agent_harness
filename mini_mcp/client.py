"""MCP 客户端管理器 ——连接 MCP 服务器，获取工具列表"""

import asyncio
import logging
from typing import List, Optional, Dict, Any

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

logger = logging.getLogger(__name__)

class MCPClientManager:
    """管理 MCP 客户端的生命周期和工具获取"""

    def __init__(self, mcp_configs: List[Dict[str, Any]]):
        self.mcp_configs = mcp_configs or []
        self.client = None
        self.lock = asyncio.Lock()

    def _build_server_config(self) -> Dict[str, Dict[str, Any]]:
        """将 mini_agent 的 MCP 配置转为 langchain-mcp-adapters 格式"""
        server_configs = {}

        for i, cfg in enumerate(self.mcp_configs):
            name = cfg.get("name") or f"mcp_server_{i+1}"
            server_cfg = {}

            # 自动检测传输类型
            url = cfg.get("connection_url", "")
            if url:
                if url.startswith("ws://") or url.startswith("wss://"):
                    server_cfg["transport"] = "websocket"
                    server_cfg["url"] = url
                elif url.startswith("http://") or url.startswith("https://"):
                    server_cfg["transport"] = "http"
                    server_cfg["url"] = url
            else:
                # stdio 模式（本地命令行启动）
                server_cfg["transport"] = "stdio"
                if cfg.get("command"):
                    server_cfg["command"] = cfg["command"]
                    server_cfg["args"] = cfg.get("args", [])
            
            if cfg.get("connection_token"):
                server_cfg["connection_token"] = cfg["connection_token"]
            if cfg.get("env"):
                server_cfg["env"] = cfg["env"]

            server_configs[name] = server_cfg

        return server_configs
    
    async def get_or_create_client(self) -> Optional[object]:
        """获取或创建 MCP 客户端实例"""

        async with self.lock:
            if self.client is not None:
                return self.client

            if not HAS_MCP:
                logger.warning("langchain-mcp-adapters 未安装，无法使用 MCP 功能")
                return None
        
            if not self.mcp_configs:
                logger.warning("未配置任何 MCP 服务器，无法创建客户端")
                return None

            try:
                server_configs = self._build_server_config()
                self.client = MultiServerMCPClient(server_configs)
                logger.info(f"MCP 客户端已创建: {len(server_configs)} 个服务器")
                return self.client
            except Exception as e:
                logger.error(f"创建 MCP 客户端失败: {e}")
                return None

    async def get_all_tools(self) -> list:
        """获取所有 MCP Server 的工具列表"""
        client = await self.get_or_create_client()
        if client is None:
            return []
        try:
            return await client.get_tools()
        except Exception as e:
            logger.warning(f"获取 MCP 工具失败: {e}")
            return []