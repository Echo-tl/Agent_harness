"""MCP 工具选择器 ——LLM 智能筛选相关工具"""

import json
import re

class MCPToolSelector:
    """用 LLM 分析 query 和工具列表，选出最相关的工具"""

    def __init__(self):
        pass

    async def select_relevant_tools(self, query: str, all_tools: list, max_tools: int = 3) -> list:
        """LLM 智能选择相关工具"""
        if not all_tools:
            return []
        if len(all_tools) <= max_tools:
            return all_tools
        
        # 构建工具描述列表
        tools_info = []
        for i, tool in enumerate(all_tools):
            tools_info.append({
                "index": i,
                "name": tool.name,
                "description": tool.description or "无描述",
            })

        # 构建 prompt
        prompt = f"""研究问题：{query}

  可用工具列表：
  {json.dumps(tools_info, ensure_ascii=False, indent=2)}

  请从以上工具中选择 {max_tools} 个与研究问题最相关的工具。
  返回格式（JSON）：
  {{
      "selected_tools": [
          {{"index": 0, "name": "tool_name", "reason": "选择原因", "relevance_score": 9}}
      ],
      "selection_reasoning": "整体选择策略说明"
  }}"""
        
        from llm_config import llm, add_cost, estimate_llm_cost
        from langchain_core.messages import HumanMessage

        try:
            response = await llm.ainvoke(
                [HumanMessage(content=prompt)],
                temperature=0.0,  # 工具选择必须确定
            )
            add_cost(estimate_llm_cost(
                input_chars=len(prompt),
                output_chars=len(response.content),
            ))

            # 解析 JSON
            parsed = self._parse_selection(response.content)
            selected = []
            for sel in parsed.get("selected_tools", []):
                idx = sel.get("index")
                if idx is not None and 0 <= idx < len(all_tools):
                    selected.append(all_tools[idx])
                    print(f"[MCP] 选择工具: {sel.get('name')} (分数: {sel.get('relevance_score')}) - {sel.get('reason')}")
            
            return selected[:max_tools]
        
        except Exception as e:
            print(f"[MCP] 工具选择失败: {e}，使用 fallback")
            return self._fallback_selection(all_tools, max_tools)

    def _parse_selection(self, text: str) -> dict:
        """尝试解析 LLM 输出的 JSON，容错处理"""
        
        # 尝试 json_repair
        try:
            import json_repair
            return json_repair.loads(text)
        except Exception:
            pass

        # 尝试正则提取 JSON
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {}
    
    def _fallback_selection(self, all_tools: list, max_tools: int) -> list:
        """关键词匹配回退：按工具名匹配 research/search/get/fetch 等关键词"""

        patterns = ['search', 'get', 'read', 'fetch', 'find', 'list', 'query', 'lookup']

        scored = []
        for tool in all_tools:
            name = tool.name.lower()
            desc = (tool.description or "").lower()
            score = sum(3 if p in name else 1 if p in desc else 0 for p in patterns)
            if score > 0:
                scored.append((score, tool))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for t, _ in scored[:max_tools]]
