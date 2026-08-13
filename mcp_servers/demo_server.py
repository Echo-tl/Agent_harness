"""MCP 测试服务器 — 暴露数学计算 + 模拟搜索工具"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MiniAgent Tools")


@mcp.tool()
def calculator(expression: str) -> str:
    """执行数学计算。参数 expression 是数学表达式字符串，如 '3.14 * 2 + 1'"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@mcp.tool()
def get_current_time() -> str:
    """获取当前系统时间"""
    from datetime import datetime
    return f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@mcp.tool()
def query_database(sql: str) -> str:
    """（模拟）执行 SQL 查询数据库"""
    return f"[模拟] SQL '{sql[:50]}...' 已执行，返回 3 条记录。"


if __name__ == "__main__":
    print("[MCP Server] 启动中...")
    mcp.run()
