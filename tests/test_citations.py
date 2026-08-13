"""引用校验模块测试 —— extract_citations / verify_citations / coverage / groundedness。"""

import pytest

from citations.verifier import extract_citations, verify_citations


def test_extract_citations_only_http():
    md = (
        "text [来源A](https://a.com/x) and [B](http://b.com/y).\n"
        "[not a link](relative/path) [mailto](mailto:x@y.z)"
    )
    cites = extract_citations(md)
    urls = [c.url for c in cites]
    assert urls == ["https://a.com/x", "http://b.com/y"]


def test_verify_citations_coverage_and_groundedness():
    md = (
        "第一段内容引用 [来源](https://a.com/x)。\n"
        "第二段内容引用 [来源2](https://b.com/y)。\n"
        "第三段没有任何引用。\n"
    )
    report = verify_citations(md, {"https://a.com/x", "https://b.com/y"})
    assert report.total_citations == 2
    assert report.grounded_citations == 2
    assert report.missing_urls == []
    assert report.coverage == pytest.approx(2 / 3)
    assert report.groundedness == 1.0


def test_verify_detects_fabricated_url():
    md = "[编造的引用](https://fake.example/not-found)"
    report = verify_citations(md, {"https://real.example/actual"})
    assert report.total_citations == 1
    assert report.grounded_citations == 0
    assert report.groundedness == 0.0
    assert report.missing_urls == ["https://fake.example/not-found"]
    # 证据集里没被用到的来源被标记出来
    assert report.uncited_evidence == ["https://real.example/actual"]


def test_verify_url_normalization_trailing_slash():
    md = "[来源](https://a.com/x/)"
    report = verify_citations(md, {"https://a.com/x"})
    assert report.grounded_citations == 1
    assert report.missing_urls == []


def test_verify_empty_report():
    report = verify_citations("", set())
    assert report.total_citations == 0
    assert report.coverage == 0.0
    assert report.groundedness == 0.0


def test_verify_report_to_dict():
    md = "[x](https://a.com/x)"
    stats = verify_citations(md, {"https://a.com/x"}).to_dict()
    assert stats["coverage"] >= 0.0
    assert stats["groundedness"] == 1.0


def test_appendix_does_not_inflate_metrics():
    """参考资料附录里的链接天然都在证据集里，不能计入正文统计。"""
    md = (
        "正文第一句引用 [来源](https://a.com/x)。\n"
        "正文第二句没有引用。\n"
        "\n"
        "## 参考资料 (Sources)\n\n"
        "- [a](https://a.com/x)\n"
        "- [b](https://b.com/y)\n"
        "- [c](https://c.com/z)\n"
    )
    report = verify_citations(md, {"https://a.com/x", "https://b.com/y", "https://c.com/z"})
    assert report.total_sentences == 2          # 只算正文两句
    assert report.total_citations == 1          # 附录链接不计入
    assert report.grounded_citations == 1
    assert report.coverage == pytest.approx(0.5)
    assert report.groundedness == 1.0
    assert report.uncited_evidence == ["https://b.com/y", "https://c.com/z"]


def test_structural_lines_not_counted_as_sentences():
    """标题/分隔线/表格分隔行是 markdown 结构，不是正文句子。"""
    md = (
        "# 报告标题\n\n"
        "## 引言\n\n"
        "这是正文 [来源](https://a.com/x)。\n\n"
        "---\n\n"
        "| 列A | 列B |\n"
        "|---|---|\n"
    )
    report = verify_citations(md, {"https://a.com/x"})
    assert report.total_sentences == 1          # 只有一句正文
    assert report.total_citations == 1
    assert report.coverage == 1.0
