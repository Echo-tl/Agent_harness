"""Tavily 检索器 ——Web 搜索"""

import asyncio
import os
from tavily import TavilyClient
from retrievers.base import BaseRetriever

class TavilyRetriever(BaseRetriever):
    def search(self) -> list[dict]:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(self.query, max_results=self.max_results)
        results = response.get("results", [])

        return [
            {
                "title": r.get("title", ""),
                "href": r.get("url", ""),
                "body": r.get("content", ""),
            }
            for r in results
        ]

    async def asearch(self) -> list[dict]:
        """TavilyClient 是同步客户端，放到线程池执行避免阻塞事件循环。"""
        return await asyncio.to_thread(self.search)

