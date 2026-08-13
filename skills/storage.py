"""Skills 加载器 —— 扫描 skills/ 目录，解析 SKILL.md frontmatter，按渐进式披露加载。

设计：
- 上下文里**只放轻量索引**（每个 skill 的 name + description），占用的 token 很少。
- 完整方法论正文**不注入上下文**，由 `use_skill(name)` 工具在模型判断任务匹配时
  按需读取返回（渐进式披露）—— 这是标准 skill 系统的做法。
- 报告节点等代码层直接按需读引用模板（get_report_citation_rules），不受此影响。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILL_MD_FILE = "SKILL.md"

# frontmatter：文件开头的 `---\n key: value\n...\n---` 块
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_META_LINE_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*$")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    path: Path


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """把 SKILL.md 拆成 (meta_dict, body)；没有 frontmatter 时 meta 为空 dict。"""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text.strip()
    meta: dict = {}
    for line in m.group(1).splitlines():
        lm = _META_LINE_RE.match(line)
        if lm:
            meta[lm.group(1)] = lm.group(2).strip()
    body = text[m.end():].strip()
    return meta, body


def load_skills(
    skills_dir: str | Path | None = None,
    enabled: list[str] | None = None,
) -> list[Skill]:
    """扫描 skills_dir 下每个子目录的 SKILL.md，返回 Skill 列表（按名称排序）。

    Args:
        skills_dir: skills 根目录；None 时用项目根下的 skills/。
        enabled: 只加载这些名称的 skill；None 或空列表表示全部加载。
    """
    root = Path(skills_dir) if skills_dir else PROJECT_ROOT / "skills"
    if not root.is_dir():
        return []

    skills = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        skill_file = child / SKILL_MD_FILE
        if not skill_file.is_file():
            continue
        text = skill_file.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)
        name = meta.get("name", child.name)
        if enabled and name not in enabled:
            continue
        skills.append(
            Skill(
                name=name,
                description=meta.get("description", ""),
                content=body,
                path=child,
            )
        )
    return skills


def load_enabled_skills() -> list[Skill]:
    """从 CONFIG["skills"] 读取 dir / enabled 并加载（enabled 为空则全部加载）。"""
    from config import CONFIG
    cfg = CONFIG.get("skills", {})
    return load_skills(
        skills_dir=cfg.get("dir", "skills"),
        enabled=cfg.get("enabled") or None,
    )


def is_skill_enabled(name: str) -> bool:
    """判断某个 skill 是否在 CONFIG["skills"]["enabled"] 里。

    enabled 未配置或为空列表时表示全部可用（返回 True）。
    use_skill 工具据此拒绝加载被禁用的 skill，防止绕过配置。
    """
    from config import CONFIG
    enabled = CONFIG.get("skills", {}).get("enabled")
    if not enabled:
        return True
    return name in enabled


def build_skills_index(skills: list[Skill]) -> str:
    """轻量索引：每个 skill 一行 `- name: description`。

    只把这份索引放进 agent 上下文（渐进式披露），完整方法论由
    `use_skill` 工具按需加载。空列表返回空串。
    """
    if not skills:
        return ""
    lines = ["<available_skills>"]
    for s in skills:
        desc = s.description or "(无描述)"
        lines.append(f"- {s.name}: {desc}")
    lines.append("</available_skills>")
    return "\n".join(lines)


def read_skill_content(skills_dir: str | Path | None, name: str) -> str | None:
    """按名称返回某个 skill 的完整方法论正文（去掉 frontmatter）。

    `use_skill` 工具用它做渐进式披露：模型需要时才从磁盘读取。
    找不到返回 None。
    """
    root = Path(skills_dir) if skills_dir else PROJECT_ROOT / "skills"
    skill_file = root / name / SKILL_MD_FILE
    if not skill_file.is_file():
        return None
    text = skill_file.read_text(encoding="utf-8")
    _, body = _parse_frontmatter(text)
    return body or None


def get_skill(skills: list[Skill], name: str) -> Skill | None:
    for s in skills:
        if s.name == name:
            return s
    return None


def get_skill_reference(
    skills_dir: str | Path | None,
    skill_name: str,
    ref: str,
) -> str | None:
    """读 skill 目录下的 references/<ref> 文件内容（如 references/apa.md）。"""
    root = Path(skills_dir) if skills_dir else PROJECT_ROOT / "skills"
    p = root / skill_name / "references" / ref
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8")


def get_report_citation_rules(
    report_format: str,
    skills_dir: str | Path | None = None,
) -> str:
    """按报告引用格式读 systematic-literature-review 的 references/<fmt>.md 模板。

    只有当该 skill 处于 enabled（或 enabled 未配置）时才用模板；
    被禁用或读不到对应模板（未知格式）时返回项目原来的内联兜底文本。
    """
    fmt = (report_format or "apa").lower()
    ref = None
    if is_skill_enabled("systematic-literature-review"):
        ref = get_skill_reference(
            skills_dir, "systematic-literature-review", f"{fmt}.md"
        )
    if ref:
        return ref.strip()
    return f"每段关键信息后使用 {fmt.upper()} 格式的文中引用，如：([来源名称](url))"
