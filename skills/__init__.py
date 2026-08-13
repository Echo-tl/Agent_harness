"""Skills 库 —— 静态加载的方法论包。

每个 skill 是 skills/<name>/SKILL.md（frontmatter + 正文），可选 references/ 模板。
不做按问题动态筛选：config 里 enabled 的 skill 在启动时全部读出，拼进 agent 上下文。
"""
