"""并发网页爬取器 ——从 URL 列表抓取全文内容"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from config import CONFIG
from runtime.context import get_run_context

class Scraper:
    """并发爬取 URL 列表，返回全文内容列表"""

    def __init__(self, urls):
        # 1. 去重（保持顺序，用 dict.fromkeys）
        self.urls = list(dict.fromkeys(urls))

        # 2. 从 config 读配置
        self.user_agent = CONFIG["scraper"]["user_agent"]
        self.timeout = CONFIG["scraper"]["timeout"]
        self.max_concurrency = CONFIG["scraper"]["max_concurrency"]
        self.min_content_length = CONFIG["scraper"]["min_content_length"]

        # 3. 并发控制信号量（最多同时爬 N 个 URL）
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

        # 4. 进度计数
        self._done = 0
        self._total = len(self.urls)

    async def _scrape_url(self, url):
        """爬取单个 URL，返回 dict: {url, title, raw_content}"""

        async with self.semaphore:       # 并发控制在这里生效
            try:
                # 设置超时 + User-Agent 
                timeout =  aiohttp.ClientTimeout(total=self.timeout)
                headers = {"User-Agent": self.user_agent}

                async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                    async with session.get(url) as response:
                        if response.status != 200:
                            return {"url": url, "title": "", "raw_content": ""}
                        
                        html = await response.text()

                # 用 BeautifulSoup 提取正文
                soup = BeautifulSoup(html, "html.parser")

                # 去掉 script、style、nav、footer 等噪音标签
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()  # 从 HTML 树里彻底删除那个标签，比 extract() 更干净
                
                title = soup.title.string if soup.title else ""
                # 提取所有文字，用换行拼接
                raw_content = soup.get_text(separator="\n", strip=True)
                
                # 过滤太短的内容（广告页、空白页）
                if len(raw_content) < self.min_content_length:
                    return {"url": url, "title": title, "raw_content": ""}
                
                return {
                    "url": url,
                    "title": title,
                    "raw_content": raw_content,
                }
            except Exception as e:
                print(f"[Scraper] 爬取失败 {url}: {e}")
                return {"url": url, "title": "", "raw_content": ""}
            finally:
                self._done += 1
                get_run_context().push_progress(f"正在爬取 {self._done}/{self._total}：{url[:60]}")

    async def run(self):
        """并发爬取所有 URL，返回有效内容列表"""
        if not self.urls:
            return []
        
        print(f"[Scraper] 开始爬取 {len(self.urls)} 个 URL (并发={self.max_concurrency})...")

        # 创建所有爬取任务
        tasks = [self._scrape_url(url) for url in self.urls] 

        # 并发执行
        results = await asyncio.gather(*tasks)

        # 过滤掉空的内容
        valid = [
            r for r in results  
            if r["raw_content"] and len(r["raw_content"]) >= self.min_content_length
        ]

        # 统计字数
        total_chars = sum(len(r["raw_content"]) for r in valid)
        print(f"[Scraper] 完成: {len(valid)}/{len(self.urls)} 个 URL 有效, 总计 {total_chars} 字")

        return valid


