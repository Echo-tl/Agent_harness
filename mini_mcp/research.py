"""MCP 研究执行器 ——LLM + MCP 工具自主研究"""

class MCPResearchSkill:
    """MCP 研究执行器：使用 LLM + MCP 工具进行自主研究"""

    async def conduct_research_with_tools(self, query: str, selected_tools: list) -> list[dict]:
        """LLM 带着 MCP 工具执行研究，返回 [{title, href, body}]"""
        if not selected_tools:
            print("[MCP] 没有可用工具，无法执行研究")
            return []
        
        from llm_config import llm, add_cost, estimate_llm_cost
        from langchain_core.messages import HumanMessage

        llm_with_tools = llm.bind_tools(selected_tools)

        prompt = f"""使用可用工具研究以下问题：{query}

  研究完成后，请总结你的发现。如果工具返回了数据，请引用具体数据。"""
        
        try:
            response = await llm_with_tools.ainvoke([HumanMessage(content=prompt)])

        except Exception as e:
            print(f"[MCP] 研究执行失败: {e}")
            return []
        
        # 处理 tool_calls 返回

        results = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc.get("name", "unknown_tool")
                tool_args = tc.get("args", {})

                # 找到对应工具并执行
                tool = next((t for t in selected_tools if t.name == tool_name), None)
                if not tool:
                    continue

                try:
                    if hasattr(tool, 'ainvoke'):
                        result = await tool.ainvoke(tool_args)
                    elif hasattr(tool, 'invoke'):
                        result = tool.invoke(tool_args)
                    else:
                        continue
                    results.append({
                          "title": f"MCP: {tool_name}",
                          "href": f"mcp://{tool_name}",
                          "body": str(result)[:5000],
                      })
                except Exception as e:
                    print(f"[MCP] 工具 {tool_name} 执行失败: {e}")

        # 也包含 LLM 自己的分析
        if hasattr(response, 'content') and response.content:
            results.append({
                "title": f"MCP 分析: {query[:50]}",
                "href": "mcp://llm_analysis",
                "body": response.content[:5000],
            })

        return results