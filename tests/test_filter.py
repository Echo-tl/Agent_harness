"""Embedding 过滤节点测试 —— 同步 embedding API 走 asyncio.to_thread，不阻塞事件循环。"""

import asyncio

from nodes import filter as filter_mod
from nodes.filter import embedding_filter_node


def test_filter_fast_path_skips_embedding():
    """总字数低于阈值走快速路径，不调用 embedding。"""
    raw = [{"url": "http://a", "title": "A", "raw_content": "很短的正文"}]
    chunks = asyncio.run(embedding_filter_node("q", raw))
    assert chunks == ["很短的正文"]


def test_filter_embedding_path_runs(monkeypatch):
    """超过阈值走 embedding 路径：同步 embedding 被 to_thread 包裹仍能工作。"""
    vec = [0.1] * 8
    monkeypatch.setattr(
        filter_mod,
        "_get_embedding",
        lambda texts: [vec] * (len(texts) if isinstance(texts, list) else 1),
    )
    # 内容要超过 fast_path_max_chars(8000) 才走 embedding 路径
    raw = [{"url": "http://a", "title": "A", "raw_content": "长正文 " * 3000}]
    chunks = asyncio.run(embedding_filter_node("q", raw))
    assert isinstance(chunks, list)
    assert chunks  # 相同向量 → 相似度 1.0 全部通过
    assert "[来源: http://a]" in chunks[0]


def test_get_embedding_batches_large_input(monkeypatch):
    """超过 batch_size 的输入自动分批，避免 HTTP 400 "batch too large"。"""
    from nodes import filter as filter_mod

    class _FakeClient:
        class _Emb:
            def __init__(self):
                self.calls = []

            def create(self, model, input):
                self.calls.append(input)
                _vec = type("V", (), {"embedding": [1.0]})
                return type("R", (), {"data": [_vec() for _ in range(len(input))]})()

        def __init__(self):
            self.embeddings = self._Emb()

    monkeypatch.setattr(filter_mod, "_embedding_client", _FakeClient())
    vecs = filter_mod._get_embedding(["text"] * 60)
    assert len(vecs) == 60                       # 返回全部向量
    calls = filter_mod._embedding_client.embeddings.calls
    assert all(len(b) <= 25 for b in calls)      # 每批 ≤ 25
    assert len(calls) == 3                       # 60 → 25+25+10
