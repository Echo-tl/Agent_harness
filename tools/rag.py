"""本地知识库 RAG —— 从 knowledge/ 目录加载 txt 到 Chroma 向量库并检索。

工程化改进
──────────
- 之前 `load_knowledge()` 在 import 时自动执行，每次启动都会重新调用
  embedding API。现在改为懒加载：首次调用 `rag_search` 时才加载，
  且已存在的 chunk_id 直接跳过，避免重复请求与启动浪费。
- 向量库路径锚定到项目根目录（不依赖运行时 cwd）。
"""

import threading
from pathlib import Path

import chromadb
from openai import OpenAI
from langchain_core.tools import tool

from config import CONFIG

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 兼容 OpenAI 协议的 embedding 客户端（DashScope/百炼等）
client = OpenAI(
    api_key=CONFIG["embedding"]["api_key"],
    base_url=CONFIG["embedding"]["base_url"],
)

def get_embedding(text: str):
    """调用 embedding API，返回向量列表"""
    db = client.embeddings.create(
        model=CONFIG["embedding"]["model"],
        input=text,
    )
    return db.data[0].embedding

chroma_client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "chroma_db"))
collection = chroma_client.get_or_create_collection("research_knowledge")

# ── 懒加载状态（线程安全）──
_loaded = False
_load_lock = threading.Lock()


def load_knowledge(knowledge_dir: str | None = None):
    """扫描 knowledge/ 目录，把 txt 切成段落存入 Chroma。

    已存在的 chunk_id 会跳过 —— 避免重启时重复调用 embedding API。
    """
    if knowledge_dir is None:
        knowledge_dir = CONFIG["rag"]["knowledge_dir"]

    knowledge_path = Path(knowledge_dir)
    if not knowledge_path.exists():
        print(f"[RAG] 目录 '{knowledge_dir}' 不存在，跳过")
        return

    files = list(knowledge_path.glob("*.txt"))
    if not files:
        print(f"[RAG] 目录 '{knowledge_dir}' 下没有 .txt 文件")
        return

    # 拿已有 id，避免重复 embedding
    existing_ids = set(collection.get(include=[])["ids"])
    print(f"[RAG] 发现 {len(files)} 个文件，已存在 {len(existing_ids)} 条向量...")

    added = 0
    for file_path in files:
        file_name = file_path.name
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        # 按空行拆成段落
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_name}_{i}"
            if chunk_id in existing_ids:
                continue
            try:
                embedding = get_embedding(chunk)
                collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{"source": file_name, "chunk": i}],
                )
                existing_ids.add(chunk_id)
                added += 1
            except Exception as e:
                print(f"[RAG] 加载失败 {chunk_id}: {e}")

    print(f"[RAG] 加载完成，新增 {added} 条，vector store 共 {collection.count()} 条记录")


def ensure_loaded(knowledge_dir: str | None = None) -> bool:
    """懒加载知识库：首次调用时才真正加载，后续直接返回。线程安全。"""
    global _loaded
    if _loaded:
        return True
    with _load_lock:
        if _loaded:
            return True
        load_knowledge(knowledge_dir)
        _loaded = True
        return True


def ensure_knowledge_loaded() -> bool:
    """显式预热入口 —— api.py 启动（lifespan）时可主动调用。"""
    return ensure_loaded()


@tool
def rag_search(query: str) -> str:
    """搜索本地知识库中的文档内容。

      适用场景：
      - 用户询问的知识、概念、定义可能在本地存储的文档里
      - 需要查找已加载的参考资料中的具体信息

      不适用场景：
      - 实时新闻、最新动态（此时应使用 web_search）
      - 和本地知识库主题明显无关的问题
    """
    try:
        ensure_loaded()
        query_vec = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=CONFIG["rag"]["top_k"],  # 在 Chroma 里找最相似的 3 个文档块
        )
    except Exception as e:
        return f"[RAG] 检索出错: {e}"

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return "[RAG] 本地知识库中未找到相关内容。"

    # zip(docs, metas) 把两个列表一对一配对，每次循环拿到一条文档 + 它对应的 metadata。
    lines = []
    for j, (doc, meta) in enumerate(zip(docs, metas)):
        src = meta.get("source", "未知") if meta else "未知"
        lines.append(f"[{j}] 来源: {src}\n{doc}")

    return "\n\n".join(lines)
