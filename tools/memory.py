

import chromadb
from langchain_core.tools import tool
from datetime import datetime
from pathlib import Path
from tools.rag import get_embedding
from config import CONFIG

# 向量库存到项目根目录的 chroma_db/（不依赖运行时 cwd）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

chroma_client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "chroma_db"))
collection = chroma_client.get_or_create_collection("user_memory")


def save_research_memory(question: str, final_report: str):
    """把一次研究的结论（字符串）打包成一条记忆，转成向量存进去"""

    memory_text = f"研究问题: {question}\n\n研究结论:\n{final_report}"

    # 长度过程直接截断
    if len(memory_text) > 2000: 
        memory_text = memory_text[:2000]

    # 向量化
    embedding = get_embedding(memory_text)
    collection.add(
        ids=[f"research_{datetime.now().isoformat()}"],
        documents=[memory_text],
        embeddings=[embedding],
        metadatas=[{"type": "research_summary", "question": question[:200], "timestamp": datetime.now().isoformat()}]
    )

    print(f"[Memory] 研究记忆已保存，vector store 共 {collection.count()} 条记录，保存名称为：{collection.name}")



@tool
def memory_search(query: str) -> str:
    """搜索用户的历史研究记录和偏好

      适用场景：
      - 用户提到之前做过的研究
      - 需要回顾过去的分析结论

      不适用场景：
      - 实时新闻、最新动态（此时应使用 web_search）
      - 和本地知识库主题明显无关的问题
    """
    try:
        query_vec = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=CONFIG["rag"]["top_k"],  # 在 Chroma 里找最相似的 3 个文档块
        )
    except Exception as e:
        return f"[Memory] 检索出错: {e}"
    
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    
    if not docs:
        return "[Memory] 本地知识库中未找到相关内容。"
    
    # zip(docs, metas) 把两个列表一对一配对，每次循环拿到一条文档 + 它对应的 metadata。
    lines = []
    for j, (doc, meta) in enumerate(zip(docs, metas)):
        time = f"时间: {meta.get('timestamp', '未知')}\n问题：{meta.get('question', '')}"
        lines.append(f"[{j}] 来源: {time}\n{doc}")
    
    return "\n\n".join(lines)