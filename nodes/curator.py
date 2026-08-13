"""Curator 节点 ——LLM 评估来源可信度和相关性，过滤低质来源"""

import json_repair
import json
from config import CONFIG
from prompts import CURATOR_SYSTEM_PROMPT

async def curate_urls(urls_and_titles: list[dict], query: str) -> list[dict]:
    """评估来源质量，过滤低质来源

      Args:
          urls_and_titles: [{url, title}, ...]
          query: 用户研究问题

      Returns:
          过滤后的 [{url, title, score, reason}, ...]，按分数降序
    """
    # 来源太少时跳过
    if len(urls_and_titles) <=2:
        return urls_and_titles
    
    sources_json = json.dumps(
        [{"url": s["url"], "title": s.get("title", "")} for s in urls_and_titles],
        ensure_ascii=False,
    )

    prompt = f"研究问题: {query}\n\n待评估来源:\n{sources_json}"

    from llm_config import llm, add_cost, estimate_llm_cost
    from langchain_core.messages import SystemMessage, HumanMessage

    temp = CONFIG["temperature"]["curator"]

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=CURATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt)],
            temperature=temp,
        )
        # 容错解析
        from utils.parsing import robust_json_parse
        evaluations = robust_json_parse(response.content)
        # 过滤 + 排序
        kept = [e for e in evaluations if e.get("keep", True)]
        kept.sort(key=lambda x: x.get("score", 0), reverse=True)
        print(f"[Curator] {len(urls_and_titles)} →{len(kept)} 个高质量来源")
        for e in kept:
            print(f"  {e['score']}/10 | {e['url'][:60]} | {e.get('reason', '')}")

        return kept
    except Exception as e:
        print(f"[Curator] 评估失败: {e}，返回全部来源")
        return urls_and_titles

