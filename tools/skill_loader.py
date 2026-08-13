"""use_skill 工具 —— 渐进式披露：模型按需加载某个 skill 的完整方法论。

可用 skill 及其一句话描述在系统提示词的 <available_skills> 索引里。
当任务匹配某个 skill 时，模型调用本工具获取详细步骤后再执行，
从而避免把全部 skill 正文常驻在上下文中。
"""

from langchain_core.tools import tool

from skills.storage import load_enabled_skills, read_skill_content, is_skill_enabled


@tool("use_skill")
def use_skill(name: str) -> str:
    """加载并返回指定 skill 的完整方法论，按它来执行当前任务。

    可用 skill 的 name 见系统提示词中的 <available_skills> 列表。
    当你的任务匹配某个 skill 时调用本工具，获取其详细步骤。
    若没有匹配的 skill，直接按常规方式处理，不要调用本工具。
    """
    # 渐进式披露 + enabled 校验：被禁用的 skill 无法绕过 config 加载
    if not is_skill_enabled(name):
        known = ", ".join(s.name for s in load_enabled_skills()) or "无"
        return f"[skill] 名为 {name} 的 skill 未启用或不存在。可用：{known}"
    content = read_skill_content(None, name)
    if content is None:
        known = ", ".join(s.name for s in load_enabled_skills()) or "无"
        return f"[skill] 未找到名为 {name} 的 skill。可用：{known}"
    return f'<skill name="{name}">\n{content}\n</skill>'
