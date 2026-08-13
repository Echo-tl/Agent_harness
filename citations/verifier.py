"""引用校验 —— 从报告正文抽取文中引用，与证据集比对，统计 coverage / groundedness。

目的：报告声称有引用，但没有 claim→source 映射的话引用就不可信。
这里把报告正文里的 `[文本](url)` 引用抽出来，与 RunContext 里记录的 evidence
（真正爬取/检索到的来源）比对，量化两件事：

- coverage（引用覆盖率）：正文句子中"带文中引用"的比例。
- groundedness（落地率）：正文引用 URL 中有多少真的落在证据集里（不是编造的）。

修正说明
────────
coverage / groundedness 只统计**正文**，不统计自动生成的"参考资料/Sources"附录：
附录里的链接天然都来自证据集，若计入会把两个指标虚高，掩盖"正文根本没引用"
的问题。coverage 以"句子"为单位（按 。！？； 与换行切分），并排除 markdown
结构行（标题、分隔线、表格分隔行），避免把格式当内容。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# markdown 链接 [label](url)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# 参考资料 / Sources 附录标题（其后都是自动生成的来源清单，不算正文）
_APPENDIX_RE = re.compile(
    r"^#{1,4}\s*.*(参考资料|参考来源|Sources|References).*$",
    re.MULTILINE | re.IGNORECASE,
)

# 报告末尾的元信息行："> 本次研究 API 花费: ..."
_COST_LINE_RE = re.compile(r"^>\s*本次研究 API 花费.*$", re.MULTILINE)


@dataclass
class Citation:
    """报告正文里的一条文中引用"""
    label: str
    url: str


@dataclass
class CitationReport:
    """引用校验结果（仅针对正文）"""
    total_sentences: int = 0
    sentences_with_citation: int = 0
    total_citations: int = 0
    grounded_citations: int = 0
    missing_urls: list[str] = field(default_factory=list)
    uncited_evidence: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """正文句子中带引用的比例 0~1"""
        if self.total_sentences == 0:
            return 0.0
        return self.sentences_with_citation / self.total_sentences

    @property
    def groundedness(self) -> float:
        """正文引用 url 落在证据集的比例 0~1"""
        if self.total_citations == 0:
            return 0.0
        return self.grounded_citations / self.total_citations

    def to_dict(self) -> dict:
        return {
            "total_sentences": self.total_sentences,
            "sentences_with_citation": self.sentences_with_citation,
            "total_citations": self.total_citations,
            "grounded_citations": self.grounded_citations,
            "coverage": round(self.coverage, 3),
            "groundedness": round(self.groundedness, 3),
            "missing_urls": self.missing_urls[:20],
            "uncited_evidence": self.uncited_evidence[:20],
        }


def extract_citations(report_md: str) -> list[Citation]:
    """抽取 markdown 报告正文里的 `[label](url)` 引用（只取 http(s)）。"""
    citations = []
    for m in _LINK_RE.finditer(report_md):
        label, url = m.group(1), m.group(2)
        if url.startswith(("http://", "https://")):
            citations.append(Citation(label=label, url=url))
    return citations


def _body_text(report_md: str) -> str:
    """去掉自动生成的参考资料附录与末尾元信息，只保留正文。

    附录标题即 `## 参考资料 (Sources)` 这类行，其后的链接清单全部剔除。
    """
    m = _APPENDIX_RE.search(report_md)
    if m:
        report_md = report_md[: m.start()]
    report_md = _COST_LINE_RE.sub("", report_md)
    return report_md


def _is_structural_line(line: str) -> bool:
    """markdown 结构行（标题 / 分隔线 / 表格行）不计入正文句子。

    表格整体是结构化数据，靠其前后的正文段落引用支撑 —— 逐行计入句子
    会不合理地稀释 coverage，所以表头/分隔/内容行都不计。
    """
    if line.startswith("#"):
        return True
    if re.fullmatch(r"[-*_=]{3,}", line):
        return True
    if line.startswith("|"):  # 表格行（表头/分隔/内容）
        return True
    return False


def _split_sentences(body: str) -> list[str]:
    """按中文句末标点（。！？；）与换行把正文切成句子。"""
    sentences = []
    for raw_line in body.split("\n"):
        line = raw_line.strip()
        if not line or _is_structural_line(line):
            continue
        parts = re.split(r"[。！？；!?]+", line)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def _normalize_url(url: str) -> str:
    """去掉尾部斜杠、fragment、query 参数，做宽松比对。"""
    url = url.strip()
    url = url.split("#")[0].split("?")[0]
    return url.rstrip("/")


def verify_citations(report_md: str, evidence_urls: set[str]) -> CitationReport:
    """把报告正文中的引用与证据集比对，返回 CitationReport。

    Args:
        report_md: 生成的 markdown 报告全文。
        evidence_urls: RunContext 里记录的证据 URL 集合（真正检索/爬取到的来源）。

    说明：
        - 只统计正文（自动生成的"参考资料/Sources"附录与末尾 cost 行不计入），
          避免附录链接把 coverage / groundedness 虚高。
        - coverage 以"句子"为单位：正文句子中带文中引用（[x](url)）的比例。
        - groundedness 是正文引用 URL 中落在证据集的比例。
    """
    body = _body_text(report_md)
    sentences = _split_sentences(body)
    citations = extract_citations(body)

    report = CitationReport()
    report.total_sentences = len(sentences)
    report.sentences_with_citation = sum(1 for s in sentences if _LINK_RE.search(s))

    evidence_norm = {_normalize_url(u) for u in evidence_urls}
    seen_urls_norm: set[str] = set()
    for c in citations:
        report.total_citations += 1
        url_norm = _normalize_url(c.url)
        if url_norm in evidence_norm:
            report.grounded_citations += 1
        elif url_norm not in seen_urls_norm:
            report.missing_urls.append(c.url)
        seen_urls_norm.add(url_norm)

    # 未被任何正文引用使用的证据（提示素材没被充分利用）
    for u in sorted(evidence_urls):
        if _normalize_url(u) not in seen_urls_norm:
            report.uncited_evidence.append(u)

    return report
