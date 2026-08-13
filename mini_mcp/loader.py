"""MCP 工具加载器 — 从 extensions_config.json 加载 MCP 工具"""

import json
import logging
from pathlib import Path
from typing import Optional

from mini_mcp.client import MCPClientManager

logger = logging.getLogger(__name__)

# 缓存：避免重复加载
_cached_tools: Optional[list] = None
_config_mtime: float = 0


def get_mcp_tools(config_path: str = "extensions_config.json") -> list:
    """加载所有启用的 MCP Server 的工具"""
    global _cached_tools, _config_mtime

    path = Path(config_path)
    if not path.exists():
        logger.warning(f"extensions_config.json 不存在: {config_path}")
        return []

    # mtime 检测：文件改了就重载
    current_mtime = path.stat().st_mtime
    if _cached_tools is not None and current_mtime == _config_mtime:
        return _cached_tools

    with open(path, encoding="utf-8") as f:
        config = json.load(f)

    servers = config.get("mcpServers", {})

    # 收集启用的 server 配置
    active_configs = []
    for name, cfg in servers.items():
        if not cfg.get("enabled", True):
            continue
        active_configs.append({"name": name, **cfg})

    if not active_configs:
        return []

    # 用同步方式加载（避开事件循环冲突）
    manager = MCPClientManager(active_configs)
    tools = _load_tools_sync(manager)

    _cached_tools = tools
    _config_mtime = current_mtime
    logger.info(f"MCP 工具加载完成: {len(tools)} 个工具")
    return tools


def _load_tools_sync(manager: MCPClientManager) -> list:
    """同步加载 MCP 工具（用子进程避开事件循环冲突）"""
    import asyncio
    import threading

    result: list = []

    def _run():
        nonlocal result
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            client = loop.run_until_complete(manager.get_or_create_client())
            if client:
                tools = loop.run_until_complete(client.get_tools())
                result = list(tools)
            loop.close()
        except Exception as e:
            logger.warning(f"MCP 工具加载失败: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=15)

    return result
