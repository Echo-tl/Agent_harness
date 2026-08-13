"""
这个节点的职责：
将"结构化摘要" + "用户问题" → 生成"最终回答"
"""

import asyncio

from state import MultiAgentState
from langchain_core.messages import SystemMessage, HumanMessage
from prompts import ANSWER_PROMPT
from tools.memory import save_research_memory
from llm_config import get_total_cost, estimate_llm_cost, add_cost
from runtime.context import get_run_context
from citations.verifier import verify_citations


async def final_report(state: MultiAgentState) -> dict:
    """回答节点：基于摘要生成最终回答

    输入：state["messages"]
    输出：{"final_report": "...", "citation_stats": {...}}

    这是 Graph 的最后一个节点，产出面向用户的最终结果。
    异步实现：所有 LLM 调用都用 ainvoke，不阻塞事件循环。
    """
    # 取用户的第一条消息作为原始问题（不是最后一条，防止被中间追问/拆分污染）
    question = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) and msg.content.strip():
            question = msg.content
            break
    if not question:
        question = state["messages"][-1].content  # 兜底

    from datetime import datetime
    current_date = datetime.now().strftime("%Y年%m月%d日")

    research_results = []
    for msg in state["messages"]:
        if hasattr(msg, "name") and msg.name == "task":
            research_results.append(msg.content)

    notes_text = "\n\n---\n\n".join(research_results) if research_results else "无搜索结果"

    # 告知 LLM 当前真实日期，防止用训练数据截止日期判断"2025是未来"
    notes_text = f"[系统提示: 当前日期是{current_date}，以下所有素材均来自实时网络搜索结果]\n\n" + notes_text

    ctx = get_run_context()

    # ── 步骤 A：生成大纲（必须先完成，章节需要它）──
    print(f"[Report] 生成大纲...")
    ctx.push_progress("正在生成报告大纲…")
    sections = await _generate_outline(question, notes_text)
    print(f"[Report] 大纲: {sections}")
    ctx.push_progress(f"大纲完成（{len(sections)} 章），正在撰写引言…")

    # ── 步骤 B/C/D 中互相独立的调用并行发起 ──
    # 引言、标题不依赖章节正文；结论依赖正文，需等正文写完。
    print(f"[Report] 生成引言...")
    ctx.push_progress("正在生成引言与标题…")
    introduction_task = _generate_introduction(question, notes_text)
    title_task = _generate_title(question, notes_text)
    introduction, title = await asyncio.gather(introduction_task, title_task)
    ctx.push_progress("引言完成，开始撰写正文…")

    # ── 步骤 C：逐章写（有依赖，须顺序）──
    all_content = []
    for i, section_title in enumerate(sections, 1):
        print(f"[Report] 正在写第 {i}/{len(sections)} 章: {section_title}")
        ctx.push_progress(f"正在撰写第 {i}/{len(sections)} 章：{section_title}")

        previous = "\n\n---\n\n".join(all_content) if all_content else ""
        section_content = await _generate_section(question, section_title, notes_text, previous)

        all_content.append(f"## {section_title}\n\n{section_content}")
        ctx.push_progress(f"第 {i}/{len(sections)} 章完成")

    # ── 合并 + 格式 ──
    body_text = "\n\n".join(all_content)

    # ── 步骤 D：生成结论（基于报告正文，不是笔记！）──
    print(f"[Report] 生成结论...")
    ctx.push_progress("正在生成结论…")
    conclusion = await _generate_conclusion(question, body_text)

    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")

    # ── 拼接最终报告 ──
    full_report = f"# {title}\n\n"
    full_report += f"> 研究日期: {date_str}  |  原始问题: {question}\n\n"
    full_report += "---\n\n"
    full_report += f"## 引言\n\n{introduction}\n\n"
    full_report += "---\n\n"
    full_report += body_text
    full_report += "\n\n---\n\n"
    full_report += f"## 结论\n\n{conclusion}"

    # ── 参考资料 + 引用校验 ──
    ctx = get_run_context()
    evidence_urls = {ev.url for ev in ctx.evidence}
    if evidence_urls:
        sources_block = "\n\n---\n\n## 参考资料 (Sources)\n\n"
        for ev in ctx.evidence:
            title_text = ev.title if ev.title else ev.url
            sources_block += f"- [{title_text}]({ev.url})\n"
        full_report += sources_block

    ctx.push_progress("正在校验引用…")
    citation_stats = verify_citations(full_report, evidence_urls).to_dict()
    print(f"[Report] 引用校验: coverage={citation_stats['coverage']}, groundedness={citation_stats['groundedness']}")

    # ── cost ──
    cost_info = f"\n\n---\n> 本次研究 API 花费: ${get_total_cost():.4f}"
    full_report += cost_info

    import os
    import re

    # 1. 确保 reports/ 目录存在
    ctx.push_progress("正在保存报告…")
    os.makedirs("reports", exist_ok=True)

    # 2. 从标题生成安全的文件名
    safe_title = re.sub(r'[\\/:*?"<>|]', '', title)[:50]   # 去掉非法字符
    filename = f"reports/{date_str}_{safe_title}.md"

    # 3. 写文件
    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"[Report] 已保存: {filename}")

    print(f"[Report] 完成: {len(sections)} 章, {len(full_report)} 字")

    # 保存研究记忆（同步 embedding + chroma，放线程池避免阻塞）
    await asyncio.to_thread(save_research_memory, question, full_report)

    return {
        "final_report": full_report,
        "total_cost": get_total_cost(),
        "citation_stats": citation_stats,
    }

def _trim_notes(notes: list[str], max_chars: int = 20000) -> list[str]:
    """从后往前保留 notes，总字符数不超过 max_chars"""

    trimmed = []
    total = 0
    for note in reversed(notes): # 倒序遍历（最新的优先保留）
        note_len = len(note)
        if total + note_len <= max_chars:
           trimmed.insert(0, note)       # 插到前面，保持原始顺序
           total += note_len
        else:
            break

    return trimmed

async def _generate_outline(question: str, notes_text: str) -> list[str]:
    """让 LLM 根据研究笔记生成报告大纲（章节标题列表）"""

    from llm_config import llm
    from langchain_core.messages import SystemMessage, HumanMessage

    prompt = f"""根据以下研究素材，为报告生成 3-5 个章节标题。
    用户问题：{question}

    研究素材：
    {notes_text[:10000]}

    要求：
    1. 每个章节标题一行，用 "1. 标题" 格式
    2. 标题要简洁（不超过 20 字）
    3. 章节之间互补不重复
    4. 只返回章节标题列表，不要其他内容"""

    response = await llm.ainvoke(
        [SystemMessage(content="你是研究报告结构设计专家。"),
        HumanMessage(content=prompt)],
        temperature=0.25,  # 低温度确保大纲稳定
    )

    # 解析：每行提取标题
    titles = []
    for line in response.content.strip().split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
           # 去掉序号前缀
           for sep in [" ", ".", "-", "、"]:
                if line[0].isdigit() and sep in line[1:3]:
                    line = line.split(sep, 1)[1].strip()
                    break

        if line:
            titles.append(line)

    return titles if titles else ["研究结果"]  # 兜底


async def _generate_section(query: str, section_title: str, notes_text: str, previous_sections: str) -> str:
    """生成单个章节的内容，告知已有内容以避免重复"""

    from llm_config import llm, add_cost, estimate_llm_cost
    from langchain_core.messages import SystemMessage, HumanMessage
    from config import CONFIG

    report_cfg = CONFIG["report"]

    # 构造"已写内容"提示（如果是第一章则跳过）
    dedup_hint = ""
    if previous_sections:
        dedup_hint = f"""
    ## ⚠️以下内容已在报告的前面章节中写过，请勿重复：
    {previous_sections[:2000]}
    请在写作时确保本章内容与上面不重复，聚焦新的角度。"""

    system_prompt = f"""你是研究报告撰写专家。请撰写报告中的一个章节。

  ## 格式要求（必须遵守）
  1. **标题层级**：章节内容使用 H3(###) 作为小节标题，不要使用 H1 或 H2
  2. **列表**：罗列项目、步骤、要点时必须使用 markdown 无序列表(- )或有序列表(1. )
  3. **表格**：呈现对比数据、统计信息、多维度比较时，必须使用 markdown 表格
  4. **高亮**：核心发现、关键数字、重要结论使用 **粗体** 标注
  5. **引用**：每条关键论断后必须用"引用格式规则"（见文末）标注文中引用，禁止编造 URL
  6. **段落**：每段不超过 5 行，不同主题之间用空行分隔
  7. **语气**：{report_cfg['tone']}
  8. **语言**：{report_cfg['language']}

  ## 禁止
  - 禁止写成长篇大论的无分隔段落墙
  - 禁止在表格中使用多行文字（保持单元格简洁）
  - 禁止编造没有来源支撑的数据

  **关键**：你的训练数据截止于2024年。当前是2026年。研究素材来自实时网络搜索。如果素材中包含2025-2026年的项目、论文、数
  据，它们真实存在。绝对不要声称"尚未发生"或"不可获取"。

  """

    # 引用格式规则单独拼接（模板里可能有 {} 花括号，不能放进上面的 f-string）
    from skills.storage import get_report_citation_rules
    citation_rules = get_report_citation_rules(report_cfg["report_format"])
    system_prompt += "\n\n## 引用格式规则\n" + citation_rules

    user_prompt = f"""用户问题：{query}
    当前章节：{section_title}
    研究素材（所有笔记，不限于本章）：
    {notes_text[:15000]}
    {dedup_hint}

    请撰写章节「{section_title}」的内容。"""

    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt),
         HumanMessage(content=user_prompt)],
        temperature=CONFIG["temperature"]["report"],
    )

    add_cost(estimate_llm_cost(
        input_chars=len(system_prompt) + len(user_prompt),
        output_chars=len(response.content),
    ))

    return response.content

async def _generate_introduction(question: str, notes_text: str) -> str:
    """基于研究素材写引言：背景铺垫 + 研究问题说明"""

    from llm_config import llm, add_cost, estimate_llm_cost
    from langchain_core.messages import SystemMessage, HumanMessage
    from config import CONFIG

    system_prompts =  f"""你是研究报告引言撰写专家。

  引言格式要求：
  - 分为 2-3 个自然段，每段不超过 5 行
  - 第一段：背景和问题重要性
  - 第二段：本报告涵盖的主要方面（用无序列表列出 3-5 个要点）
  - 语言：{CONFIG['report']['language']}
  - 不要写"本报告将..."之类的套话，直接进入主题"""

    user_prompt = f"""研究问题：{question}
    研究素材摘要：
    {notes_text[:8000]}

    请撰写报告的引言部分（200-400 字）。"""

    response = await llm.ainvoke(
        [SystemMessage(content=system_prompts),
         HumanMessage(content=user_prompt)],
        temperature=CONFIG["temperature"]["introduction"],  # 0.25，需要稳定
    )

    add_cost(estimate_llm_cost(
        input_chars=len(system_prompts) + len(user_prompt),
        output_chars=len(response.content),
    ))

    return response.content

async def _generate_conclusion(question: str, report_body: str) -> str:
    """基于已生成的报告正文写结论，不是基于原始素材"""

    from llm_config import llm, add_cost, estimate_llm_cost
    from langchain_core.messages import SystemMessage, HumanMessage
    from config import CONFIG

    system_prompt = f"""你是研究报告结论撰写专家。

  结论格式要求：
  - 第一段：总结核心发现（2-3 句话，用 **粗体** 标注关键结论）
  - 第二段：使用无序列表列出 3-5 个具体发现要点
  - 第三段（可选）：局限性或未来方向
  - 语言：{CONFIG['report']['language']}
  - 不要引入报告正文中没有的新信息"""

    user_prompt = f"""研究问题：{question}

  以下为已生成的报告正文：
  ---
  {report_body[:10000]}
  ---

  请基于以上报告正文撰写结论（200-400 字）。不要引入报告正文中没有的新信息。"""

    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt),
         HumanMessage(content=user_prompt)],
        temperature=CONFIG["temperature"]["conclusion"],  # 0.25，结论要准确
    )

    add_cost(estimate_llm_cost(
        input_chars=len(system_prompt) + len(user_prompt),
        output_chars=len(response.content),
    ))

    return response.content

async def _generate_title(question: str, notes_text: str) -> str:
    """基于研究素材生成报告标题"""

    from llm_config import llm, add_cost, estimate_llm_cost
    from langchain_core.messages import SystemMessage, HumanMessage

    prompt = f"""请基于以下研究问题和素材，为研究报告生成一个简洁、有信息量的标题。

  用户原始问题: {question}

  研究素材摘要:
  {notes_text[:3000]}

  要求:
  1. 标题必须直接回应用户的原始问题，不能偏离到其他领域
  2. 不要超过 30 字
  3. 使用中文
  4. 只返回标题文本，不要带引号、序号或其他格式"""

    response = await llm.ainvoke(
        [SystemMessage(content="你是研究报告标题撰写专家。"),
         HumanMessage(content=prompt)],
        temperature=0.25,
    )

    add_cost(estimate_llm_cost(
        input_chars=len(prompt),
        output_chars=len(response.content),
    ))

    return response.content.strip().strip('"').strip("'").strip("《").strip("》")
