"""Arxiv 检索器 ——学术论文搜索"""

import asyncio
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from retrievers.base import BaseRetriever

class ArxivRetriever(BaseRetriever):
    """用 Arxiv 官方 API 搜索论文（免费，无需 API Key）"""

    BASE_URL = "https://export.arxiv.org/api/query"

    def search(self) -> list[dict]:
        # 构造查询 URL
        params = {
            "search_query": f"all:{self.query}",
            "max_results": str(self.max_results),
            "sortBy": "relevance",
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                xml_data = resp.read().decode("utf-8")
            
            # 解析 XML 响应
            root = ET.fromstring(xml_data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            results = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                link = entry.find("atom:id", ns)

                results.append({
                    "title": title.text.strip() if title is not None else "",
                    "href": link.text.strip() if link is not None else "",
                    "body": summary.text.strip()[:500] if summary is not None else "",
                })
            return results
        except Exception as e:
            print(f"[Arxiv] 搜索失败: {e}")
            return []

    async def asearch(self) -> list[dict]:
        """urllib 是同步阻塞 IO，放到线程池执行避免阻塞事件循环。"""
        return await asyncio.to_thread(self.search)