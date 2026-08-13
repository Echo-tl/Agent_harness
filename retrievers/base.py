"""检索器基类 ——定义统一接口"""

import asyncio


class BaseRetriever:
    """所有检索器必须实现 search() 方法，返回统一格式。

    异步接口 asearch() 默认用 asyncio.to_thread 包住同步 search()，
    避免阻塞事件循环。真正的异步检索器（如 MCP）可覆盖实现。
    """

    def __init__(self, query: str, max_results: int = 5):
        self.query = query
        self.max_results = max_results

    def search(self) -> list[dict]:
        """返回 [{title, href, body}, ...] 统一格式

        title: 标题
        href:  URL
        body:  摘要/内容片段
        """
        raise NotImplementedError

    async def asearch(self) -> list[dict]:
        """异步搜索：默认把同步 search 放到线程池，不阻塞事件循环。"""
        return await asyncio.to_thread(self.search)
