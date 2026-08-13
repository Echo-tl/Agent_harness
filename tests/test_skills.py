"""Skills 加载器测试 —— frontmatter 解析 / 加载 / 拼接 / 引用模板。"""

import pytest

from skills.storage import (
    _parse_frontmatter,
    load_skills,
    load_enabled_skills,
    build_skills_index,
    read_skill_content,
    is_skill_enabled,
    get_skill,
    get_skill_reference,
    get_report_citation_rules,
)


@pytest.fixture
def skills_dir(tmp_path):
    """构造一个两 skill 的 fixture skills 目录。"""
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    (alpha / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Alpha skill\n---\n# Alpha\nDo alpha things.\n",
        encoding="utf-8",
    )
    (beta / "SKILL.md").write_text(
        "---\nname: beta\ndescription: Beta skill\n---\n# Beta\nDo beta things.\n",
        encoding="utf-8",
    )
    (alpha / "references").mkdir()
    (alpha / "references" / "apa.md").write_text("APA rules here", encoding="utf-8")
    return tmp_path


def test_parse_frontmatter():
    meta, body = _parse_frontmatter(
        "---\nname: x\ndescription: y\n---\nhello\nworld"
    )
    assert meta == {"name": "x", "description": "y"}
    assert body == "hello\nworld"


def test_parse_frontmatter_without_frontmatter():
    meta, body = _parse_frontmatter("just body")
    assert meta == {}
    assert body == "just body"


def test_load_skills(skills_dir):
    skills = load_skills(skills_dir)
    assert [s.name for s in skills] == ["alpha", "beta"]
    assert skills[0].description == "Alpha skill"
    assert "Do alpha things." in skills[0].content


def test_load_skills_enabled_filter(skills_dir):
    skills = load_skills(skills_dir, enabled=["beta"])
    assert [s.name for s in skills] == ["beta"]


def test_load_skills_missing_dir(tmp_path):
    assert load_skills(tmp_path / "nope") == []


def test_build_skills_index_is_lightweight(skills_dir):
    """渐进式披露：索引只含 name+description，不含完整正文。"""
    index = build_skills_index(load_skills(skills_dir))
    assert "- alpha: Alpha skill" in index
    assert "- beta: Beta skill" in index
    # 完整方法论不在索引里，需 use_skill 按需加载
    assert "Do beta things." not in index


def test_build_skills_index_empty():
    assert build_skills_index([]) == ""


def test_read_skill_content(skills_dir):
    content = read_skill_content(skills_dir, "alpha")
    assert "Do alpha things." in content
    assert read_skill_content(skills_dir, "nope") is None


def test_use_skill_tool_progressive_disclosure():
    """模型按需调用 use_skill 拿到完整方法论；未知 skill 返回错误。"""
    from tools.skill_loader import use_skill

    ok = use_skill.invoke({"name": "deep-research"})
    assert "Deep Research" in ok
    assert '<skill name="deep-research">' in ok

    missing = use_skill.invoke({"name": "no-such-skill"})
    assert "[skill]" in missing


def test_is_skill_enabled_respects_config(monkeypatch):
    import config as config_mod
    from skills.storage import is_skill_enabled

    monkeypatch.setitem(config_mod.CONFIG["skills"], "enabled", ["deep-research"])
    assert is_skill_enabled("deep-research") is True
    assert is_skill_enabled("academic-paper-review") is False


def test_use_skill_rejects_disabled(monkeypatch):
    """被禁用的 skill 不能被 use_skill 绕过 config 加载。"""
    import config as config_mod
    from tools.skill_loader import use_skill

    monkeypatch.setitem(config_mod.CONFIG["skills"], "enabled", ["deep-research"])
    out = use_skill.invoke({"name": "academic-paper-review"})
    assert "[skill]" in out
    assert "deep-research" in out  # 错误信息里列出可用项
    assert "Paper Review" not in out  # 没泄露禁用 skill 的正文


def test_get_report_citation_rules_respects_enabled(monkeypatch):
    """SLR skill 被禁用时，报告引用规则回退到内联兜底文本（不读模板）。"""
    import config as config_mod
    from skills.storage import get_report_citation_rules

    monkeypatch.setitem(config_mod.CONFIG["skills"], "enabled", ["deep-research"])
    rules = get_report_citation_rules("apa")
    assert "APA" in rules
    assert "参考文献列表" not in rules  # 模板独有标记不在 → 走的是兜底
    assert "([来源名称](url))" in rules


def test_get_skill(skills_dir):
    skills = load_skills(skills_dir)
    assert get_skill(skills, "alpha").name == "alpha"
    assert get_skill(skills, "nope") is None


def test_get_skill_reference(skills_dir):
    assert get_skill_reference(skills_dir, "alpha", "apa.md") == "APA rules here"
    assert get_skill_reference(skills_dir, "alpha", "missing.md") is None


def test_load_enabled_skills_reads_config():
    """真实项目里 enabled 的 skill 都能加载出来（skills/ 目录随项目附带）。"""
    skills = load_enabled_skills()
    names = {s.name for s in skills}
    assert "deep-research" in names
    assert "systematic-literature-review" in names


def test_get_report_citation_rules_real_apa():
    """真实项目随附的 APA 模板被读到（不是兜底文本）。"""
    rules = get_report_citation_rules("apa")
    assert "APA" in rules
    assert "文中引用" in rules


def test_get_report_citation_rules_fallback_for_unknown():
    rules = get_report_citation_rules("nope-format")
    assert "NOPE-FORMAT" in rules  # 兜底文本带上格式名


def test_get_report_citation_rules_uses_fixture(skills_dir):
    # 让 SLR skill 在 fixture 里存在，验证按给定目录读取
    slr = skills_dir / "systematic-literature-review"
    (slr / "references").mkdir(parents=True)
    (slr / "SKILL.md").write_text(
        "---\nname: systematic-literature-review\ndescription: x\n---\nbody\n",
        encoding="utf-8",
    )
    (slr / "references" / "apa.md").write_text("FIXTURE APA", encoding="utf-8")
    assert get_report_citation_rules("apa", skills_dir=skills_dir) == "FIXTURE APA"
