"""Retriever 工厂 ——配置驱动的多数据源选择"""

from config import CONFIG
from retrievers.tavily_retriever import TavilyRetriever
from retrievers.arvix_retriever import ArxivRetriever
from retrievers.mcp_retriever import MCPRetriever
# 注册表：配置里的名字 →类

RETRIEVER_REGISTRY = {
    "tavily": TavilyRetriever,
    "arxiv": ArxivRetriever,
    "mcp": MCPRetriever,
}

def get_retrievers(query: str) -> list:
    """从 config 读取检索器列表，返回实例列表

    config 里配 "retrievers": ["tavily", "arxiv"]
    -> 返回 [TavilyRetriever(query), ArxivRetriever(query)]
    """

    names = CONFIG["search"].get("retrievers", ["tavily"])
    instances = []
    for name in names:
        cls = RETRIEVER_REGISTRY.get(name)
        if cls:
           instances.append(cls(query=query, max_results=CONFIG["search"]["max_results"]))
        else:
           print(f"[Retriever] 未知检索器: {name}，跳过") 
    return instances

