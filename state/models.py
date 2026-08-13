"""Pydantic 模型 —— 定义结构化输出格式"""

from pydantic import BaseModel, Field

class Evidence(BaseModel):
    """一条来源证据 —— 把结论里的关键论断绑定到可点击、可校验的原始来源。

    在压缩阶段由 LLM 从搜索结果（文本里带 [来源: url] 前缀）中提取，
    后续由 citations/verifier.py 校验报告引用与证据是否一致。
    """
    source_id: str = Field(description="证据唯一 ID，通常取 URL 本身")
    url: str = Field(description="来源的可点击 URL")
    title: str = Field(default="", description="来源标题或网页标题")
    quote: str = Field(default="", description="支撑该论断的原文摘录，一两句话")
    published_at: str | None = Field(
        default=None, description="来源发布日期（如有），格式 YYYY-MM-DD"
    )

class ResearchNote(BaseModel):
    """单条研究发现"""
    topic: str = Field(description="该发现所属的主题/方面")
    key_finding: str = Field(description="核心发现，用一句话概括")
    details: str = Field(description="支持该发现的详细信息和证据")
    evidences: list[Evidence] = Field(
        default_factory=list,
        description="支撑本条发现的来源证据列表，每条必须可点击可校验",
    )

class CompressedResearch(BaseModel):
    """压缩后的研究笔记 —— compress 节点的输出"""
    summary: str = Field(description="对所有发现的总体概括，2-3 句话")
    notes: list[ResearchNote] = Field(description="各条研究发现，按主题组织")

class ConductResearch(BaseModel):
    """Call this tool to dispatch a research task to a researcher sub-agent."""
    research_topic: str = Field(description="The topic to research. Should be specific and detailed, at least a paragraph.")


class ResearchComplete(BaseModel):
    """Call this tool when enough research has been collected to write the final report."""

class ResearchReflection(BaseModel):
    """对单条研究结果的反思评估"""
    clarity_score: int = Field(
        description="结构清晰度评分 1-5。研究结果是否层次分明、逻辑清晰"
    )
    conciseness_score: int = Field(
        description="简洁度评分 1-5。是否没有废话、每句话都有信息量"
    )
    reliability_score: int = Field(
        description="可靠性评分 1-5。是否引用了具体来源和数据，没有编造"
    )
    overall_pass: bool = Field(
        description="三个维度是否都 >= 3。是则 True，否则 False"
    )
    feedback: str = Field(
        description="如果不通过，给 Researcher 的补充搜索建议（一句话）。通过则留空"
    )

class ClarifyResult(BaseModel):
    """LLM 判断用户问题是否需要补充说明"""
    need_clarification: bool = Field(
        description="是否需要用户进一步说明"
    )
    question: str = Field(
        description="追问的内容，用 markdown 格式，可直接展示给用户"
    )