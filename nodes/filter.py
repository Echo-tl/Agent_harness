"""Embedding 过滤层 ——文本切块 + 向量相似度过滤，剔除噪音
    爬回来的全文里面混着广告、导航栏、评论区等噪音。embedding
    过滤层把文本切成小块，用余弦相似度剔除和查询不相关的 chunk，只保留最精华的 ~3000 字送给 LLM。
"""

import asyncio

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from config import CONFIG

# 从 config 读配置，创建共享的 embedding 实例
_embedding_config = CONFIG["embedding"]
_embedding_client = OpenAI(
      base_url=_embedding_config["base_url"],
      api_key=_embedding_config["api_key"],
)

def _get_embedding(text):
    """调用 DashScope embedding API，返回向量列表。

    自动按 batch_size 分批：百炼 API 单次最多 25 条，超过会报
    HTTP 400 "batch size is invalid"（爬取的文本块常远超此数）。
    """
    if isinstance(text, str):
        text = [text]
    batch_size = _embedding_config.get("batch_size", 25)
    vectors = []
    for i in range(0, len(text), batch_size):
        batch = text[i : i + batch_size]
        resp = _embedding_client.embeddings.create(
              model=_embedding_config["model"],
              input=batch,
          )
        vectors.extend(d.embedding for d in resp.data)
    return vectors

def cosine_similarity(vec_a, vec_b):
    """计算两个向量的余弦相似度，返回 0~1 之间的值"""
    a = np.array(vec_a)
    b = np.array(vec_b)
    # 点积 / (模长乘积)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def embedding_filter_node(query, raw_contents, threshold=None):
    """对爬取到的全文内容做 embedding 过滤

        参数:
          query: str —用户的研究主题（用于计算相关性）
          raw_contents: list[dict] —scraper 返回的 [{url, title, raw_content}, ...]
          threshold: float | None —相似度阈值，低于此值的 chunk 丢弃。
                    如果传 None，就用 config 里的默认值

        返回:
          list[str] —top-N 最相关的文本 chunk
    """

    # 读取配置
    chunk_size = CONFIG["filter"]["chunk_size"]
    chunk_overlap = CONFIG["filter"]["chunk_overlap"]
    if threshold is None:
        threshold = CONFIG["filter"]["similarity_threshold"]
    max_chunks = CONFIG["filter"]["max_chunks"]
    fast_path_max = CONFIG["filter"]["fast_path_max_chars"]

    if not raw_contents:
        return []
    
    # ── 2. 快速路径：总字数少就跳过过滤
    total_chars = sum(len(c["raw_content"]) for c in raw_contents)
    if total_chars < fast_path_max:
        print(f"[Filter] 总字数 {total_chars} < {fast_path_max}，跳过过滤")
        return [c["raw_content"] for c in raw_contents]
    
    # 切块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # 切分优先级
    )

    all_chunks = []
    for doc in raw_contents:
        chunks = splitter.split_text(doc["raw_content"])
        for chunk in chunks:
            all_chunks.append({
                "chunk": chunk,
                "url": doc["url"],
                "title": doc.get("title", ""),
            })
    print(f"[Filter] 切块完成: {len(all_chunks)} chunks (来自 {len(raw_contents)} 篇文档)")

    # ── 4. 获取 query 的 embedding（同步 embedding API → 线程池，不阻塞事件循环）──
    query_embedding = (await asyncio.to_thread(_get_embedding, query))[0]

    # ── 5. 获取所有 chunk 的 embedding ──
    chunk_texts = [c["chunk"] for c in all_chunks]
    chunk_embeddings = await asyncio.to_thread(_get_embedding, chunk_texts)

    # ── 6. 计算相似度，过滤 ──
    scored = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_emb)
        if score > threshold:
            scored.append((score, all_chunks[i]))

    # ── 7. 按分数从高到低排序，取 top-N ──
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:max_chunks]

    print(f"[Filter] 完成: {len(all_chunks)} chunks →{len(scored)} 个通过阈值 →保留 top {len(top_chunks)}")

    # 返回纯文本列表
    result = []
    for score, item in top_chunks:
        # 把来源信息加到 chunk 前面
        text =  f"[来源: {item['url']}]\n{item['chunk']}"
        result.append(text)

    return result

