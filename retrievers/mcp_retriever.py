"""MCP 检索器 ——接入 MCP 工具作为数据源"""

from retrievers.base import BaseRetriever
from config import CONFIG
from mini_mcp import MCPClientManager, HAS_MCP, MCPToolSelector, MCPResearchSkill

class MCPRetriever(BaseRetriever):
    def __init__(self, query: str, max_results: int = 5):
        super().__init__(query, max_results)
        self.mcp_configs = CONFIG.get("mcp", {}).get("servers", [])

    def search(self) -> list[dict]:
        """同步入口（被 BaseRetriever 接口要求）"""

        import asyncio
        import threading
        if not HAS_MCP or not self.mcp_configs:
            print("[MCP] langchain-mcp-adapters 未安装或无可用配置，无法使用 MCP 检索器")
            return []

        # 独立线程跑 MCP，避开主线程事件循环冲突
        result: list[dict] = []
        def _run():
            nonlocal result
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self._async_search())
                loop.close()
            except Exception as e:
                print(f"[MCP] 检索失败: {e}")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=30)
        return result     

    async def asearch(self) -> list[dict]:
        """异步搜索：直接走原生 MCP 异步链路。"""
        return await self._async_search()

    async def _async_search(self) -> list[dict]:
        """异步搜索入口，使用 MCP 工具进行检索"""
        mmanager = MCPClientManager(self.mcp_configs)
        selector = MCPToolSelector()
        researcher = MCPResearchSkill()
        
        # 阶段 1：获取所有工具
        all_tools = await mmanager.get_all_tools()
        if not all_tools:
            print("[MCP] 没有可用工具，无法执行检索")
            return []
        print(f"[MCP] 发现 {len(all_tools)} 个工具")

        # 阶段 2：LLM 选择相关工具
        selected = await selector.select_relevant_tools(self.query, all_tools, max_tools=3)
        if not selected:
            print("[MCP] 没有选择任何工具，无法执行检索")
            return []
        print(f"[MCP] 选中 {len(selected)} 个工具: {[t.name for t in selected]}")

        # 阶段 3：执行研究
        results = await researcher.conduct_research_with_tools(self.query, selected)
        print(f"[MCP] 研究完成: {len(results)} 条结果")

        return results