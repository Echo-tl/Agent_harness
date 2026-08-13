# Mini-DeerFlow 改造计划

> 基于 DeerFlow 源码学习总结，将 mini-research-agent 从 Workflow Agent 升级为 Long-Running Multi-Agent 项目。

---

## 当前架构 vs 目标架构

| 维度 | 当前 (mini-agent) | 目标 (Mini-DeerFlow) |
|------|------------------|---------------------|
| Agent 创建 | 手写 StateGraph + 节点 + 边 | `create_agent()` 自动生成 ReAct 循环 |
| 行为注入 | 硬编码在节点函数里 | Middleware 链（可插拔 hook） |
| 多 Agent | Supervisor 子图 + Researcher 子图 | Lead Agent + task() 工具委派 |
| 工具注册 | 手动 `ToolNode([...])` | `get_available_tools()` 流水线组装 |
| 运行时 | `main.py` 直接 `graph.ainvoke()` | Runtime 层（stream + cancel + status） |
| 上下文 | 手动 compress 节点 | SummarizationMiddleware 自动压缩 |
| 安全 | 无 | LoopDetection / TokenBudget / Guardrail |
| 沙盒 | 无 | File/Bash 工具 |
| MCP | 部分（手写 client） | 标准 MCP 集成 |
| 记忆 | 无 | Memory 系统（用户偏好持久化） |
| 人机交互 | 手动 interrupt + while 循环 | ClarificationMiddleware 自动中断 |
| 子任务委派 | 无 | SubagentExecutor + 线程池 |

---

## Level 1：用 `create_agent()` 替换手写 ReAct

### 问题

当前 `graph/__init__.py` 手写了 ReAct 循环：

```python
# 现在（手写）:
builder.add_node("research_agent", agent_node)
builder.add_node("tools", ToolNode([search, rag_search]))
builder.add_conditional_edges("research_agent", router, ...)
builder.add_edge("tools", "research_agent")
```

### 改进

用 LangChain 的 `create_agent()` 替换：

```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[search, rag_search, memory_search],
    middleware=[],
    system_prompt="你是研究员...",
)
```

核心循环由框架管理，只需关注 behavior。

### 涉及文件
- `graph/__init__.py` — 重写 `build_research_subgraph()`
- `nodes/agent.py` — 可能不再需要（被 `create_agent` 内置 model 节点替代）

---

## Level 2：引入 Middleware 机制

### 核心思想

> ReAct 循环保持简单，行为通过 Middleware 注入。

### 可加的 Middleware

```python
from langchain.agents.middleware import AgentMiddleware

# 1. 上下文压缩
class SummarizationMiddleware(AgentMiddleware):
    """对话太长时自动压缩旧消息"""
    def before_model(self, state, runtime):
        if token_count > 32000:
            return compress_old_messages(state)

# 2. 循环检测
class LoopDetectionMiddleware(AgentMiddleware):
    """防止 LLM 重复搜索同一关键词"""
    def after_model(self, state):
        if repeated_tool_calls >= 5:
            return strip_tool_calls(state)

# 3. 动态上下文
class DynamicContextMiddleware(AgentMiddleware):
    """注入当前日期"""
    def before_agent(self, state, runtime):
        return {"messages": [SystemMessage(f"今天是 {datetime.now()}")]}

# 4. 工具错误兜底
class ToolErrorHandlingMiddleware(AgentMiddleware):
    """工具崩了不崩整个 run"""
    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except Exception as e:
            return ToolMessage(content=f"Error: {e}", tool_call_id=...)
```

### Hook 类型速查

| Hook | 执行时机 | 用途 |
|------|---------|------|
| `before_agent` | Run 开始时（一次） | 注入日期、初始化 |
| `before_model` | 每次调 LLM 前 | 注入上下文、压缩 |
| `after_model` | 每次 LLM 返回后 | 批量检查 tool_calls |
| `after_agent` | Run 结束时（一次） | 清理、记忆提取 |
| `wrap_model_call` | LLM 调用前后（嵌套） | 包装 LLM 调用 |
| `wrap_tool_call` | 工具执行前后（嵌套） | 拦截/兜底 |

### 涉及文件
- 新建 `middleware/` 目录
- `graph/__init__.py` — 在 `create_agent(middleware=[...])` 中注册

---

## Level 3：加 Runtime 层

### 问题

当前 `main.py` 直接调 `graph.ainvoke()`——无流式、无取消、无状态追踪。

### 改进

加轻量 Runtime 层：双mode还是实现，（取消控制 + 状态追踪）对于 mini 项目还没实现

```python
# mini_worker.py
async def run_agent(graph, input, config, stream_callback):
    """Agent 的运行时外壳"""
    try:
        await stream_callback({"type": "metadata", "run_id": run_id})

        async for chunk in graph.astream(input, config,
                                          stream_mode=["values","messages"]):
            if cancelled.is_set():
                break
            await stream_callback(chunk)

        status = "success" if not cancelled.is_set() else "interrupted"
    except Exception as e:
        status = "error"
    finally:
        await stream_callback({"type": "end"})
```

### 涉及文件
- 新建 `runtime/worker.py`
- `main.py` — 改为调用 worker

---

## Level 4：Supervisor-Worker 工具委派模式

### 问题

当前 Supervisor 是图节点，Researcher 是子图。两者耦合在图结构里。

### 改进

把 Researcher 变成工具。Supervisor 通过 `task()` 委派：

```python
# task_tool — 委派研究任务
@tool("task")
async def task_tool(description: str, prompt: str) -> str:
    """委派研究任务给 Researcher"""
    executor = SubagentExecutor(tools=[search, rag_search], ...)
    task_id = executor.execute_async(prompt)

    # 轮询等待完成
    while True:
        result = get_task_result(task_id)
        if result.status == COMPLETED:
            return f"研究完成: {result.result}"
        await asyncio.sleep(5)

# Supervisor Agent
supervisor = create_agent(
    model=llm,
    tools=[task_tool, write_report],   # ← task() 只是另一个工具
    middleware=[...],
)
```

### 涉及文件
- 新建 `subagents/executor.py`
- `nodes/supervisor.py` — 改为工具
- `graph/__init__.py` — 简化图结构

---

## Level 5：加 File/Bash Sandbox

### 新增工具

```python
# 文件工具
read_file(path: str) -> str
write_file(path: str, content: str) -> None
ls(dir: str) -> str

# Shell 工具
bash(command: str) -> str
```

### Sandbox 设计要点（参考 DeerFlow）

- **虚拟路径**：Agent 看到 `/mnt/user-data/workspace/`，物理映射到本地目录
- **安全限制**：可选禁用 `host_bash`，或放 Docker 隔离
- **输出裁剪**：bash 输出超过阈值 → 截断 + 引用文件

```python
class LocalSandbox:
    def execute_command(self, command):
        result = subprocess.run(command, shell=True, capture_output=True)
        return result.stdout[:MAX_CHARS]  # 裁剪输出

    def read_file(self, path):
        return Path(real_path).read_text()

    def write_file(self, path, content):
        Path(real_path).parent.mkdir(parents=True, exist_ok=True)
        Path(real_path).write_text(content)
```

### 涉及文件
- 新建 `sandbox/` 目录
- `tools/__init__.py` — 注册 sandbox 工具

---

## Level 6：MCP 集成

### 当前状态

已有 `mini_mcp/` 目录和手写 MCP client，但未标准化。

### 改进

参考 DeerFlow 模式：

```python
# extensions_config.json
{
  "mcpServers": {
    "github": {
      "enabled": true,
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    }
  }
}

# 加载 MCP 工具
from mini_mcp.client import MCPClient
mcp_tools = MCPClient.load_tools(config)
all_tools = config_tools + builtin_tools + mcp_tools
```

### 参考 DeerFlow

- 懒加载 + mtime 缓存失效
- 支持 stdio/SSE/HTTP 三种传输
- `tool_search.enabled` — 可选延迟加载 MCP 工具 schema

---

## Level 7：Long-Running 机制

| 能力 | DeerFlow | Mini 版实现 |
|------|---------|------------|
| 上下文压缩 | SummarizationMiddleware | token > 阈值 → 调 LLM 压缩旧消息 |
| 用户记忆 | MemoryMiddleware + memory.json | JSON 文件存偏好，下次对话注入 |
| 人机交互 | ClarificationMiddleware | `ask_clarification` → Command(goto=END) |
| 死循环检测 | LoopDetectionMiddleware | tool_call 重复 3 次 → 警告，5 次 → 强制停止 |
| Token 预算 | TokenBudgetMiddleware | 总数 > 阈值 → 强制结束 |

---

## 建议实施顺序

```
第 1 步：用 create_agent() 重写 Researcher           ← 1-2 天
         - graph/__init__.py 简化
         - 移除手写 router/agent_node

第 2 步：加第一批 middleware                           ← 1 天
         - Summarization（压缩）
         - LoopDetection（防死循环）
         - ToolErrorHandling（兜底）

第 3 步：Supervisor → task() 工具委派                 ← 1 天
         - 新建 subagents/executor.py
         - supervisor 从图节点变成 tool

第 4 步：加 Runtime 层                                ← 1 天
         - 新建 runtime/worker.py
         - 流式输出 + 取消 + 状态追踪

第 5 步：加 File/Bash Sandbox                         ← 1 天
         - 新建 sandbox/
         - read_file / write_file / ls / bash

第 6 步：MCP 标准化 + 长对话机制                       ← 1-2 天
         - 标准化 MCP 集成
         - 加 Memory / Clarification
```

---

## 架构目标

改造后的目标架构：

```
                    ┌── runtime/worker.py ──┐
用户输入 ──────────→│  stream + cancel +    │──→ SSE → 前端
                    │  status tracking      │
                    └────────┬──────────────┘
                             │
                    ┌────────▼──────────────┐
                    │   create_agent()       │
                    │   + middleware 链       │
                    │   + tools 流水线        │
                    └────────┬──────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         bash/read/     search/rag/     task(researcher)
         write_file     web_fetch       │
         (sandbox)      (MCP)           │
                                   ┌────▼────────┐
                                   │ SubagentExecutor
                                   │ 线程池 + _aexecute
                                   │ checkpointer=False
                                   └─────────────┘
```

## 与 DeerFlow 的差异

| | DeerFlow | Mini-DeerFlow |
|--|---------|---------------|
| Middleware | 28 个 | 5-8 个核心的 |
| Subagent 类型 | general-purpose / bash / 自定义 | researcher / bash |
| 并发 | 线程池(3) + 独立 event loop | 单线程池 |
| 存储 | SQLite/Postgres | JSON 文件 + SQLite checkpoint |
| 前端 | Next.js | 命令行 / 简单 API |
| IM | Slack/Telegram/飞书... | 无 |
| Auth | SSO/OIDC | 无 |
