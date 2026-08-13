"""Evidence / ResearchNote 结构化模型测试。"""

from state.models import Evidence, ResearchNote, CompressedResearch


def test_evidence_defaults():
    ev = Evidence(source_id="s1", url="https://x.example")
    assert ev.quote == ""
    assert ev.title == ""
    assert ev.published_at is None


def test_research_note_evidences_default_empty():
    note = ResearchNote(topic="t", key_finding="k", details="d")
    assert note.evidences == []


def test_research_note_with_evidences():
    note = ResearchNote(
        topic="t",
        key_finding="k",
        details="d",
        evidences=[
            Evidence(source_id="s1", url="https://x.example", quote="quote"),
        ],
    )
    assert note.evidences[0].url == "https://x.example"


def test_compressed_research_roundtrip():
    note = ResearchNote(topic="t", key_finding="k", details="d")
    obj = CompressedResearch(summary="s", notes=[note])
    data = obj.model_dump()
    assert data["notes"][0]["evidences"] == []


def test_evidence_missing_url_fails():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Evidence(source_id="s1")  # url 必填
