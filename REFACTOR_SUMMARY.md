# Mini-DeerFlow 改造记录：Level 1-4

> 从 2026-07-08 到 2026-07-09，基于 DeerFlow 源码学习，对 mini-research-agent 进行架构升级。

---

## 改造总览

| Level | 内容 | 状态 |
|-------|------|------|
| 1 | `create_agent()` 替换手写 ReAct | ✅ |
| 2 | 4 个 Middleware | ✅ |
| 3 | `astream` 流式输出 | ✅ |
| 4 | Supervisor → `task()` 工具委派 | ✅ |

---

## Level 1：用 `create_agent()` 替换手写 ReAct

### 改了什么

**Researcher 子图**：从手写 StateGraph（agent_node + router + ToolNode + compress_node）改成 `create_agent()` + 外层包装 compress。

### 删除的代码

| 文件 | 删除内容 |
|------|---------|
| `graph/__init__.py` | `router()` 函数（第 46-59 行，手写路由逻辑） |
| `nodes/agent.py` | `agent_node()` 函数（手写 model 节点，第 18-59 行） |

### 修改的代码

**`state/__init__.py`** — ResearcherState 字段改名：

```python
# 改前
class ResearcherState(TypedDict):
    research_messages: Annotated[list[BaseMessage], add_messages]  # ← 旧名

# 改后
class ResearcherState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # ← create_agent 标准字段名
```

**`nodes/compress.py`** — `research_messages` → `messages`

**`nodes/supervisor_tools.py`** — `research_messages` → `messages`

**`graph/__init__.py`** — `build_research_subgraph()`：

```python
# 改前（手写 ReAct）
def build_research_subgraph():
    builder = StateGraph(ResearcherState, output=ResearcherOutputState)
    builder.add_node("research_agent", agent_node)
    builder.add_node("tools", ToolNode([search, rag_search, memory_search]))
    builder.add_conditional_edges("research_agent", router, ...)
    builder.add_edge("tools", "research_agent")
    builder.add_node("compress", compress_node)
    ...

# 改后（create_agent + 外层包装）
from langchain.agents import create_agent

def build_research_subgraph():
    agent = create_agent(
        model=llm,
        tools=[search, rag_search, memory_search],
        system_prompt=SYSTEM_PROMPT,
        state_schema=ResearcherState,
        middleware=[...],
    )
    builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
    builder.add_node("agent", agent)      # ← create_agent 返回值作为子图节点
    builder.add_node("compress", compress_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "compress")
    builder.add_edge("compress", END)
    return builder.compile()
```

### 核心认知

`create_agent()` 自动做了三件事：
1. 构建消息列表（`system_prompt=` 参数）
2. 绑定工具 + 调 LLM（`tools=` 参数）
3. 路由决策（有 tool_calls → tools；无 → END）

不再需要手写 `agent_node` 和 `router`。

---

## Level 2：加入 4 个 Middleware

### 新建文件

```
middleware/
  __init__.py
  summarization.py          # 上下文压缩
  tool_error_handling.py    # 工具兜底
  loop_detection.py         # 死循环检测
  dynamic_context.py        # 注入当前日期
```

### Middleware 1：SummarizationMiddleware

- **Hook**：`before_model`（每次调 LLM 前）
- **逻辑**：检查 messages 总字符数 → 超过阈值 → 调 LLM 压缩旧消息 → 返回摘要 + 保留最近消息
- **阈值**：8000 字符（测试用 3000）

```python
class SummarizationMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        messages = state.get("messages", [])
        total_chars = sum(len(m.content) for m in messages if hasattr(m, "content"))
        if total_chars // 2 < 8000:
            return None      # 不触发
        # 压缩逻辑：分割消息 → 调 LLM 总结 → 返回新 messages
```

### Middleware 2：ToolErrorHandlingMiddleware

- **Hook**：`wrap_tool_call`（包裹每个工具执行）
- **逻辑**：try/except 包裹 `handler(request)` → 崩了返回 ToolMessage 错误信息

```python
class ToolErrorHandlingMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except Exception as e:
            return ToolMessage(content=f"工具执行出错: {e}", tool_call_id=...)

    async def awrap_tool_call(self, request, handler):  # async 版本（ainvoke 需要）
        try:
            return await handler(request)
        except Exception as e:
            return ToolMessage(content=f"工具执行出错: {e}", tool_call_id=...)
```

### Middleware 3：LoopDetectionMiddleware

- **Hook**：`after_model`（LLM 返回后）
- **逻辑**：hash 每个 tool_call → 存 history → 同一 hash 出现 5 次 → 清空 tool_calls，强制结束

```python
class LoopDetectionMiddleware(AgentMiddleware):
    def __init__(self):
        self.history = []   # 最近 20 轮 tool_call hash

    def after_model(self, state, runtime):
        for tc in last_message.tool_calls:
            tool_hash = hash(tc["name"] + str(tc["args"]))
            self.history.append(tool_hash)
            if len(self.history) > 20:
                self.history = self.history[-20:]
            if self.history.count(tool_hash) >= 5:
                return {"messages": [...清空 tool_calls 的 AIMessage...]}
```

### Middleware 4：DynamicContextMiddleware

- **Hook**：`before_agent`（run 开始时，只一次）
- **逻辑**：检查是否已有 `<system-reminder>` → 没有则注入当前日期

```python
class DynamicContextMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        for msg in state["messages"]:
            if "<system-reminder>" in str(msg.content):
                return None   # 已注入 → 不重复
        current_date = datetime.now().strftime("%Y-%m-%d, %A")
        return {"messages": [HumanMessage(f"<system-reminder>当前日期: {current_date}"), ...]}
```

### 核心认知

| Hook | 时机 | 用途 |
|------|------|------|
| `before_agent` | Run 开始时（一次） | 注入日期、初始化 |
| `before_model` | 每次调 LLM 前 | 压缩、注入上下文 |
| `after_model` | 每次 LLM 返回后 | 检测循环、批量检查 |
| `wrap_tool_call` | 每次工具执行前后 | 兜底、拦截 |

Middleware 不是工具，是**图上的钩子**。`create_agent()` 自动发现 hook → 创建节点 → 连线。

---

## Level 3：astream 流式输出

### 改了什么

`main.py`：从 `graph.ainvoke()` 改为 `graph.astream(stream_mode="values")`。

```python
# 改前（阻塞）
result = await graph.ainvoke(initial_state, config)
print(result["final_report"])

# 改后（流式）
async for chunk in graph.astream(initial_state, config, stream_mode="values"):
    last_state = chunk
    print(f"当前 messages 数量: {len(chunk.get('messages', []))}")
print(last_state["final_report"])
```

### stream_mode 对比

| mode | 返回 | 用途 |
|------|------|------|
| `"values"` | 每次 super-step 的完整 state | 拿 final_report、追踪状态 |
| `"updates"` | 每次 super-step 的增量变化 | 只了解本轮变了什么 |
| `"messages"` | 新增的消息对象 | 流式 token 展示 |

### 核心认知

- `ainvoke` = 等全部完成 → 返回
- `astream` = 每步 yield → 可以实时展示进度、检测取消、记录状态

---

## Level 4：Supervisor → `task()` 工具委派

### 改了什么

Supervisor 从**图节点**改成 **Agent**。Researcher 从**子图**改成**工具**。

### 新建文件

- `subagents/executor.py` — `task_tool` 定义

### 修改的文件

| 文件 | 修改 |
|------|------|
| `nodes/supervisor.py` | 删掉 `supervisor_node`、`ConductResearch`；改用 `create_agent(...)` |
| `graph/__init__.py` | 删掉 `build_supervisor_subgraph()`、`reflection_node` |
| `subagents/executor.py` | 新文件：`@tool("task")` 工具 |

### 删除的文件

- `nodes/supervisor_tools.py` — 逻辑被 `task_tool` + `create_agent()` 替代

### 架构对比

```
改前（图结构）:
  clarify → planner → supervisor → final_report
                         │
                    supervisor_subgraph (内部):
                      supervisor_node (手动调 LLM)
                      supervisor_tools_node (调 build_research_subgraph)
                      reflection_node (评估质量)
                         │
                    再回到 supervisor_node（Command goto）


改后（工具委派）:
  clarify → planner → supervisor_agent → final_report
                         │
                    create_agent(model, tools=[task_tool, ResearchComplete])
                         │
                    LLM 自主决定:
                      task("研究A") → Researcher 子图 → 返回结果
                      task("研究B") → Researcher 子图 → 返回结果
                      够了 → ResearchComplete()
```

### task_tool 核心代码

```python
# subagents/executor.py
@tool("task")
async def task_tool(research_topic: str) -> str:
    """委派研究任务给 Researcher 子 Agent"""
    researcher = build_research_subgraph()
    result = await researcher.ainvoke({
        "messages": [HumanMessage(content=research_topic)],  # ← 关键！
        "research_topic": research_topic,
    })
    return result["compressed_research"]
```

### 为什么加了 `HumanMessage` 才正常

`create_agent()` 只看 `state["messages"]`，不知道 `state["research_topic"]`。如果不把研究主题放进 `messages`，LLM 不知道要搜索什么 → 不调工具 → 空结果 → LLM 用训练数据"编"答案。

### 为什么 reflection_node 不需要了

LLM 在 ReAct 循环中自己判断"够了"：看到研究结果 → 觉得不够 → 继续调 `task()` → 够了 → 调 `ResearchComplete()`。不需要外部节点评估。

### 为什么 Supervisor 不需要多轮手动调 LLM

`create_agent()` 内置 ReAct：model → tool_calls → tools → model → ... 直到 LLM 不给 tool_calls。不再需要 `supervisor_node → supervisor_tools_node → reflection_node → 再回到 supervisor_node`。

---

## 改造后的完整架构

```
用户输入
  │
  ▼
graph.astream() — 流式执行
  │
  ├── clarify (节点) — 追问澄清
  ├── planner (节点) — 生成研究计划
  │
  ├── supervisor_agent (create_agent) — 研究总指挥
  │     │
  │     ├── task_tool("研究A") → build_research_subgraph()
  │     │     ├── create_agent(model, tools=[search, rag_search, memory_search],
  │     │     │                  middleware=[Summarization, LoopDetection,
  │     │     │                              ToolErrorHandling, DynamicContext])
  │     │     │     → ReAct: model ↔ tools
  │     │     ├── compress_node — 结构化笔记
  │     │     └── 返回 "compressed_research"
  │     │
  │     ├── task_tool("研究B") → ... 同上
  │     └── ResearchComplete() → 结束
  │
  └── final_report (节点) — 生成 markdown 报告
```

---

## 故障排查记录

### Bug 1：`messages=0` — 日志显示 middleware 读到 0 条消息

**原因**：`task_tool` 里初始 state 没有 `HumanMessage`，`create_agent` 的初始 `messages` 只有 system_prompt。

**修复**：`initial_state["messages"] = [HumanMessage(content=research_topic)]`

### Bug 2：`compress_node` 读到 `results length: 5 chars`

**原因**：同上——LLM 没搜，没有 ToolMessage。

**修复**：同上。

### Bug 3：`NotImplementedError: awrap_tool_call`

**原因**：`ToolErrorHandlingMiddleware` 只写了同步版本 `wrap_tool_call`，`ainvoke` 需要异步版本。

**修复**：加 `async def awrap_tool_call(self, request, handler)`

### Bug 4：`builder.add_edge("planner", "planner")`

**原因**：手误，自己连自己。

**修复**：改成 `builder.add_edge("clarify", "planner")`

### Bug 5：`middleware=[SummarizationMiddleware]` 传了类而不是实例

**原因**：缺少括号。

**修复**：改为 `middleware=[SummarizationMiddleware()]`

---

## 当前文件结构

```
mini_deer-flow_agent/
├── graph/
│   └── __init__.py          ← build_research_subgraph() + build_parent_graph()
├── middleware/
│   ├── summarization.py     ← 上下文压缩
│   ├── tool_error_handling.py ← 工具兜底
│   ├── loop_detection.py    ← 死循环检测
│   └── dynamic_context.py  ← 注入日期
├── nodes/
│   ├── clarify.py           ← 追问澄清
│   ├── planner.py           ← 生成研究计划
│   ├── supervisor.py        ← create_agent(model, tools=[task_tool, ResearchComplete])
│   ├── compress.py          ← 结构化笔记
│   └── report.py            ← 生成 markdown
├── subagents/
│   └── executor.py          ← @tool("task") 定义
├── state/
│   ├── __init__.py          ← ResearcherState, SupervisorState, MultiAgentState
│   └── models.py            ← Pydantic 结构化输出模型
├── main.py                  ← 入口（astream 流式）
└── IMPROVEMENT_PLAN.md      ← 完整改造计划（Level 1-7）
```

### 已删除文件

- `nodes/agent.py` — 被 `create_agent()` 替代
- `nodes/supervisor_tools.py` — 被 `task_tool` + `create_agent` 替代

---

---

## 关键时刻：bug 排查记录（2026-07-10）

改造后完整图运行了数小时无法正常工作：Supervisor 调 `task()` 但 Researcher 从未执行，`[Search]` 日志从不出现。

### 排查过程

| 步骤 | 发现 | 结论 |
|------|------|------|
| 独立测 Researcher | `build_research_subgraph().ainvoke()` 正常搜索 | Researcher 代码正确 |
| 独立测 Supervisor | `supervisor_agent.ainvoke()` 不调 task | 问题在 Supervisor |
| 加 ToolMessage 日志 | `name=None`，内容是错误信息 | 工具调用异常 |
| 换 qwen-plus | 相同问题 | 不是模型问题 |
| 换 deepseek-chat | 相同问题 | 不是模型问题 |
| 最简 `create_agent` | DeepSeek 正常调工具 | `create_agent` 本身正确 |

### 根因

**`ClarificationMiddleware` 只实现了同步版 `wrap_tool_call`，缺少异步版 `awrap_tool_call`。**

`Supervisor` 通过 `ainvoke`（异步）运行，LangGraph 的 ToolNode 需要调用 middleware 的 `awrap_tool_call`。如果只有同步版，会抛出 `NotImplementedError`：

```
Asynchronous implementation of awrap_tool_call is not available.
```

`ToolErrorHandlingMiddleware` 捕获这个异常后返回 `ToolMessage` 错误，Supervisor 的 LLM 看到错误就重试 → 反复失败 → 最终降级为 `ResearchComplete` → 直接生成报告。

### 修复

给 `ClarificationMiddleware` 加 `async def awrap_tool_call(self, request, handler)`。`ToolErrorHandlingMiddleware` 已经有了，不需要改。

### 教训

每个 `wrap_tool_call` 必须同时实现同步和异步版本。如果 Agent 用 `ainvoke/astream`（异步），LangGraph 调的是异步版；用 `invoke/stream`（同步），调的是同步版。

---

## 实际完成进度

```
✅ Level 1: create_agent() 替换手写 ReAct
✅ Level 2: 4 个 Middleware (Summarization, ToolError, LoopDetection, DynamicContext)
✅ Level 3: astream 流式输出
✅ Level 4: Supervisor → task() 工具委派 + Clarify 改 middleware
✅ Level 5: Sandbox 工具 (read_file, write_file, ls, bash)
⬜ Level 6: MCP 标准化
⬜ Level 7: 长对话机制 (Memory, TokenBudget, 取消控制, 状态追踪)
```

### Level 5 实现细节

**新建文件**：`sandbox/tools.py`

四个工具函数，所有操作限制在 `./workspace/` 目录下：

| 工具 | 签名 | 功能 |
|------|------|------|
| `read_file` | `(path: str) -> str` | 读文件，路径逃逸检测 |
| `write_file` | `(path: str, content: str) -> str` | 写文件，自动创建父目录 |
| `ls` | `(dir_path: str = ".") -> str` | 列目录 |
| `bash` | `(command: str) -> str` | 执行 shell 命令，10s 超时，5k 截断 |

**安全机制**：`_safe_path()` 用 `Path.resolve()` 消除 `..` 后检查是否在 workspace 内，防止路径逃逸。

### Level 4 关键代码回顾

**`subagents/executor.py`** — `task_tool`：
```python
@tool("task")
async def task_tool(research_topic: str) -> str:
    """委派研究任务给 Researcher 子 Agent"""
    from graph import build_research_subgraph  # 延迟导入，打破循环依赖
    researcher = build_research_subgraph()
    result = await researcher.ainvoke({
        "messages": [HumanMessage(content=research_topic)],
        "research_topic": research_topic,
    })
    return result["compressed_research"]
```

**关键设计决策**：
- `from graph import ...` 放在函数内部（延迟导入），打破 `graph → supervisor → executor → graph` 循环依赖
- 传给 Researcher 的 `messages` 必须包含 `HumanMessage(content=research_topic)`，因为 `create_agent` 只读 `messages` 字段，不读 `research_topic`

### State 清理

删掉了不再使用的字段：

| 删除 | 原因 |
|------|------|
| `MultiAgentState.supervisor_messages` | 无人写入 |
| `MultiAgentState.notes` | 改用 `messages` 中的 ToolMessage |
| `MultiAgentState.raw_notes` | 同上 |
| `MultiAgentState.visited_urls` | 模块级变量替代 |
| `SupervisorState` 整个类 | `build_supervisor_subgraph` 已删除 |
| `ResearcherState.tool_call_iterations` | `create_agent` 内部管理 |
| `override_reducer` | 无人使用 |

### Middleware 分配策略

| Middleware | Supervisor | Researcher | 原因 |
|-----------|-----------|-----------|------|
| Summarization | ✅ | ✅ | 两者都需要压缩 |
| ToolErrorHandling | ✅ | ✅ | 两者都需要兜底 |
| DynamicContext | ✅ | ✅ | 两者都需要日期 |
| Clarification | ✅ | ❌ | 只有 Supervisor 追问用户 |
| LoopDetection | ❌ | ✅ | Supervisor 调 task() 多次是正常行为 |
| TokenBudget | ✅ (30K) | ✅ (20K) | 两者都需要防止 token 溢出。Researcher 阈值更低（爬取数据多） |

### 配置要点

- **使用 deepseek-chat**：qwen-plus / qwen3.6-flash 在 `create_agent` 中工具调用不稳定。deepseek-chat 表现最稳定且价格低
- **base_url**：`https://api.deepseek.com/v1`（OpenAI 兼容格式，`ChatOpenAI` 可直接使用）
- **sandbox 工具必须有 docstring**：否则 `create_agent()` 报错 `Function must have a docstring`
- **embedding 仍用 DASHSCOPE_API_KEY**：embedding 单独走阿里云 text-embedding-v1（与 LLM 不同服务商）

### 新增文件

| 文件 | 作用 |
|------|------|
| `middleware/token_budget.py` | Token 预算保护：超限自动停止工具调用 |
| `middleware/clarification.py` | 人机交互：`interrupt()` 冻结图等用户回复（已补 `awrap_tool_call`） |
| `middleware/summarization.py` | 上下文压缩：`before_model` 检查 token，超阈值用 LLM 压缩 |
| `middleware/tool_error_handling.py` | 工具兜底：try/except 包裹工具执行，崩了返回 ToolMessage |
| `middleware/loop_detection.py` | 死循环检测：`after_model` 检查 tool_call hash，≥5 次强制停止 |
| `middleware/dynamic_context.py` | 动态上下文：`before_agent` 注入当前日期到消息 |
| `sandbox/tools.py` | 沙盒工具：`read_file`, `write_file`, `ls`, `bash` |
| `subagents/executor.py` | 子 Agent 工具：`@tool("task")` 委派研究任务 |

### 版本兼容性提醒

- **Python 3.14**：pydantic v1 不兼容，显示 `UserWarning`。不影响运行
- **每个 `wrap_tool_call` 必须同时实现 `awrap_tool_call`（异步版）**：`ainvoke` 调异步版，缺了就崩
- **`@tool` 装饰器要求 docstring 必须是函数体第一条语句**：print 放在 docstring 前面会导致 `__doc__` 为空 → 装饰器报错
- **延迟导入打破循环依赖**：`task_tool` 内部 `from graph import build_research_subgraph`
