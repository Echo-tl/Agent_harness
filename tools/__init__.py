"""
Tools 模块 —— Agent 可调用的外部能力
"""

import asyncio

from langchain_core.tools import tool
from tools.rag import rag_search
from tools.memory import memory_search
from scraper.scraper import Scraper
from nodes.filter import embedding_filter_node
from nodes.curator import curate_urls
from runtime.context import get_run_context

# 说明：访问过的 URL 不再用模块级变量 _visited_urls。
# 每个 thread_id 的访问记录、成本、证据都放在 RunContext（runtime/context.py），
# 通过 contextvar 在请求内传递，避免并发请求互相污染。


@tool
async def search(query: str) -> str:
    """搜索互联网，爬取全文，embedding 过滤后返回高质量内容。

      与简单 snippet 不同，本工具会：
      1. 用 Tavily 等检索器并发搜索最多 5 个相关网页
      2. 并发爬取每个网页的完整正文
      3. 用 embedding 过滤，只保留和 query 最相关的内容
    """

    ctx = get_run_context()
    ctx.push_progress(f"正在搜索：{query[:80]}")

    try:
        # ── 步骤 1: 多检索器并发搜索（异步，不阻塞事件循环）──
        from retrievers import get_retrievers
        retrievers = get_retrievers(query)
        gathered = await asyncio.gather(
            *(r.asearch() for r in retrievers),
            return_exceptions=True,
        )
        all_results = []
        for retriever, results in zip(retrievers, gathered):
            if isinstance(results, Exception):
                print(f"[Search] {retriever.__class__.__name__} 失败: {results}")
                continue
            all_results.extend(results)
            print(f"[Search] {retriever.__class__.__name__}: {len(results)} 条结果")
            ctx.push_progress(f"已获取 {retriever.__class__.__name__} {len(results)} 条结果")

        if not all_results:
            return "未找到相关搜索结果。"

        # ── 步骤 2: 提取 URL 并去重（跳过本 run 已访问过的）──
        urls = []
        for r in all_results:
            url = r.get("href")
            if url and url not in urls and url not in ctx.visited_urls:
                urls.append(url)

        print(f"[Search] 找到 {len(urls)} 个唯一 URL，开始并发爬取...")

        # 组装 URL+标题列表
        urls_and_titles = [
            {"url": r.get("href"), "title": r.get("title", "")}
            for r in all_results
            if r.get("href") and r.get("href") in urls  # 只评估去重后的 URL
        ]

        # LLM 评估
        ctx.push_progress(f"正在评估 {len(urls_and_titles)} 个来源的可信度…")
        curated = await curate_urls(urls_and_titles, query)

        # 用 curator 返回的 URL 列表替换 urls
        curated_urls = [c["url"] for c in curated if c.get("url")]
        if curated_urls:
            urls = curated_urls

        # 爬取前先标记——即使爬失败也不重试（本 run 内去重）
        for u in urls:
            ctx.visited_urls.add(u)

        # ── 步骤 3: 并发爬取全文 ──
        scraper = Scraper(urls)
        full_contents = await scraper.run()

        # arxiv 爬取失败 → 用 API 摘要兜底
        result_map = {r["href"]: r for r in all_results if r.get("href")}
        for c in full_contents:
            if not c.get("raw_content") and c.get("url") in result_map:
                c["raw_content"] = result_map[c["url"]].get("body", "")

        valid_count = sum(1 for c in full_contents if c["raw_content"])
        print(f"[Search] 爬取完成: {len(full_contents)} 篇，{valid_count} 篇有效内容")
        ctx.push_progress(f"搜索完成，{valid_count} 篇有效来源")

        if not valid_count:
            # 回退：至少返回 Tavily 的 snippet，同时记录证据
            formatted = []
            for i, r in enumerate(all_results, 1):
                if r.get("href"):
                    ctx.add_evidence(
                        url=r["href"],
                        title=r.get("title", ""),
                        quote=(r.get("body") or "")[:200],
                    )
                formatted.append(
                    f"[{i}] {r['title']}\n"
                    f"    链接: {r['href']}\n"
                    f"    摘要: {r['body']}"
                )
            return "\n\n".join(formatted)

        # ── 步骤 3.5: 记录来源证据（供压缩阶段的 evidence 与引用校验使用）──
        for c in full_contents:
            if c.get("raw_content"):
                ctx.add_evidence(
                    url=c["url"],
                    title=c.get("title", ""),
                    quote=c["raw_content"][:200],
                )

        # ── 步骤 4: Embedding 过滤 ──
        ctx.push_progress(f"正在向量过滤 {len(full_contents)} 篇内容…")
        filtered_chunks = await embedding_filter_node(
            query=query,
            raw_contents=full_contents,
        )
        if not filtered_chunks:
            return "爬取成功但未找到与问题相关的内容。"

        # ── 步骤 5: 格式化返回 ──
        formatted = []
        for i, chunk in enumerate(filtered_chunks, 1):
            formatted.append(f"[Chunk {i}]\n{chunk}")

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"搜索出错: {str(e)}"
