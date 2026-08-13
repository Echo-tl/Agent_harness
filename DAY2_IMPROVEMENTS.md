# 2026-07-02 改进记录

> 日期：2026-07-02
> 主要内容：MCP 集成 + 全链路 Bug 修复 + 报告质量优化

---

## 一、完成的改进项

### Step 9：MCP 集成 ✅

| 文件 | 说明 |
|------|------|
| `mini_mcp/__init__.py` | 包入口，检测 `langchain-mcp-adapters` 是否可用 |
| `mini_mcp/client.py` | MCP 客户端管理器，连接 MCP 服务器、获取工具列表 |
| `mini_mcp/tool_selector.py` | LLM 智能选择 2-3 个最相关工具 |
| `mini_mcp/research.py` | LLM + MCP 工具执行研究，返回 `[{title, href, body}]` |
| `retrievers/mcp_retriever.py` | MCP 检索器，实现 `BaseRetriever` 接口 |
| `retrievers/__init__.py` | 注册 `"mcp"` 到 `RETRIEVER_REGISTRY` |
| `config.py` | 新增 `"mcp"` 配置段 |

**MCP 工作流**：
```
MCP 服务器启动 → MCPClientManager 连接 → 发现工具列表
  → MCPToolSelector (LLM) 选 2-3 个相关工具
  → MCPResearchSkill (LLM + selected tools) 执行研究
  → 返回统一格式 → 合并到 search 结果管道
```

**命名冲突修复**：目录从 `mcp/` 改名为 `mini_mcp/`，避免和 Python 官方 `mcp` 包冲突。

---

### Step 10：报告 Markdown 格式优化

**参考 GPT-Researcher 的 prompt**，在 `nodes/report.py` 的 `_generate_section` / `_generate_introduction` / `_generate_conclusion` 中加了格式指令：

- 标题层级（H2/H3）
- Markdown 表格（对比数据、统计信息）
- 列表和粗体高亮
- 文中引用超链接 `([来源](url))`
- 段落长度限制（每段不超过 5 行）

---

## 二、今天修的 Bug（按发现顺序）

### Bug 1：`load_dotenv()` 加载太晚 🔴

**现象**：启动时报 `Missing credentials`。
**根因**：`load_dotenv()` 在 `main.py` 第 13 行才执行，但 `config.py` 的 `os.getenv("DASHSCOPE_API_KEY")` 在 import 阶段就已经跑了。
**修复**：把 `load_dotenv()` 移到 `config.py` 的最顶部（第 1-4 行）。

---

### Bug 2：Clarify 脑补"招聘""岗位" 🔴

**现象**：用户问 "帮我找 GitHub AI agent 项目和论文"，Clarify 追问 "2025-2026 年在职的岗位"。
**根因**：LLM 看到 "agent" 联想到职业/就业，prompt 没有禁止脑补。
**修复**：在 `CLARIFY_PROMPT` 中加"追问铁则"——只能追问用户消息中已提及的话题，并加了反例示范。

---

### Bug 3：Reflection 制造死循环 🔴

**现象**：Supervisor 被 Reflection 反复打回重搜，直到 `max_supervisor_iterations=5` 强制结束。
**根因**：Arxiv 论文摘要没有 URL 引用，Reflection 判为 reliability=1 → 打回 → 重搜还是同样结果 → 无限循环。
**修复 1**：`REFLECTION_PROMPT` 放宽 reliability 标准，Arxiv 论文标题也算有效来源。
**修复 2**：`nodes/reflection.py` 加 `research_iterations >= max_supervisor_iterations` 时强制结束。
**修复 3**：config 的 `max_supervisor_iterations` 从 5 降到 3，`max_researcher_tool_calls` 从 3 降到 2。

---

### Bug 4：Checkpoint 污染 🔴

**现象**：改了 UUID 后仍生成和当前问题无关的报告（"AI Agent 岗位能力要求分析"）。
**根因**：`main.py` 里 `thread_id = "research_thread_1"` 是写死的，LangGraph 从 `checkpoints.db` 加载了旧会话的状态。
**修复**：改为 `thread_id = str(uuid.uuid4())`，每次运行全新会话。

---

### Bug 5：Planner 拿到错误的问题 🔴

**现象**：research_brief 是基于 "是指AI Agent，包含RAG" 这样的追问回答片段生成的，完全偏离原始问题。
**根因**：`planner_node` 用 `state["messages"][-1]` 取最后一条消息，经过 Clarify 追问后最后一条是用户的追问回答片段。
**修复**：改为取第一条 `HumanMessage`（用户的原始问题）。

---

### Bug 6：final_report 拿到错误的问题 🔴

**现象**：报告标题和内容与用户原始问题不相关。
**根因**：同上——`final_report` 也用 `state["messages"][-1]`。
**修复**：同上——取第一条 `HumanMessage`。

---

### Bug 7：`_generate_title` prompt 没传 question 和 notes 🔴

**现象**：报告标题生成 "AI驱动的医疗影像诊断准确率评估"——和 AI Agent 完全无关。
**根因**：prompt 模板里没有 `{question}` 和 `{notes_text}`，LLM 没有上下文，纯随机猜标题。
**修复**：prompt 中加入 `{question}` 和 `{notes_text[:3000]}`。

---

### Bug 8：`json_repair.dumps` 不存在 🔴

**现象**：报告中记录 "`AttributeError: module 'json_repair' has no attribute 'dumps'`"，所有搜索失败。
**根因**：`nodes/curator.py` 调了 `json_repair.dumps()`，但 `json_repair` 只有 `loads` 没有 `dumps`。
**修复**：改为标准库 `json.dumps`。

---

### Bug 9：`OpenAIEmbeddings` 和 DashScope 不兼容 🔴

**现象**：报告中记录 "`'contents is neither str nor list of str.: payload.input.contents'` (HTTP 400)"。
**根因**：`langchain_openai.OpenAIEmbeddings` 发的请求格式和 DashScope embedding API 不兼容。
**修复**：`nodes/filter.py` 改用原生 `openai.OpenAI` 客户端（和 `tools/rag.py` 一致）。

---

### Bug 10：LLM 不信任搜索结果中的 2025-2026 年数据 🔴

**现象**：报告反复写"2025–2026 年尚未发生"、"数据不可获取"，即使搜索工具能找到真实内容。
**根因**：`qwen-plus` 训练数据截止 2024 年。LLM 把训练数据的认知（"2025 不存在"）置于搜索结果之上。
**修复**：在 `COMPRESS_PROMPT` 和报告生成 prompt 中加入："你的训练数据已过时。当前是 2026 年。搜索结果来自实时网络。如果搜索结果中有 2025-2026 年的内容，它们真实存在。"

---

## 三、关键设计原则总结

### 1. 数据管道的 Key 名一致性

不同组件返回不同 key 名，消费者必须用**生产者的 key 名**：

| 数据来自 | key 格式 |
|---------|---------|
| Retriever | `{title, href, body}` |
| Curator | `{url, score, reason, keep}` |
| Scraper | `{url, title, raw_content}` |

### 2. State 中的消息取用

`state["messages"][-1]` 取的是**最后一条消息**，经过多轮对话后被中间消息污染。取用户原始问题应用 `[0]`（第一条 HumanMessage）。

### 3. LLM 的信息优先级

LLM 天然更信训练数据而非上下文。遇到训练数据过时的问题时，需要在 prompt 中**显式声明训练数据已过时、上下文才是权威来源**。

### 4. LangGraph Checkpoint

写死 `thread_id` 会导致旧状态污染新对话。每次新研究应生成唯一 ID。

---

## 四、当前项目文件结构

```
mini_research_agent/
├── config.py              # 全局配置（含 LLM、MCP、report、temperature 等）
├── llm_config.py          # LLM 统一创建 + cost 追踪
├── main.py                # CLI 入口
├── streaming.py           # SSE 流式输出
├── api.py                 # FastAPI 端点
├── requirements.txt       # 依赖
├── .env                   # API Key
│
├── graph/__init__.py      # LangGraph 工作流编排
├── state/
│   ├── __init__.py        # MultiAgentState / SupervisorState / ResearcherState
│   └── models.py          # Pydantic 结构化输出模型
├── prompts/__init__.py    # 所有 System Prompt 模板
│
├── nodes/
│   ├── agent.py           # Researcher Agent（agent ⇄ tools 循环）
│   ├── clarify.py         # 需求澄清追问
│   ├── compress.py        # 搜索结果压缩为结构化笔记
│   ├── curator.py         # LLM 来源可信度评估
│   ├── filter.py          # Embedding 过滤层
│   ├── planner.py         # 研究计划生成
│   ├── reflection.py      # 研究质量评估
│   ├── report.py          # final_report（大纲→intro→body→conclusion）
│   ├── supervisor.py      # 研究总指挥
│   └── supervisor_tools.py # Supervisor 工具执行
│
├── tools/
│   ├── __init__.py        # search 工具（多检索器 + curator + scraper + filter 管道）
│   ├── rag.py             # 本地知识库检索
│   └── memory.py          # 历史研究记忆
│
├── retrievers/
│   ├── __init__.py        # Retriever 工厂 + RETRIEVER_REGISTRY
│   ├── base.py            # BaseRetriever 接口
│   ├── tavily_retriever.py
│   ├── arxiv_retriever.py
│   └── mcp_retriever.py   # MCP 检索器
│
├── scraper/scraper.py     # 并发网页爬取器
├── utils/parsing.py       # robust_json_parse（4 层回退）
├── mini_mcp/              # MCP 集成（原名 mcp/，避免与官方 mcp 包冲突）
│   ├── __init__.py
│   ├── client.py          # MCP 客户端管理器
│   ├── tool_selector.py   # LLM 工具选择器
│   └── research.py        # LLM + MCP 工具研究
│
├── mcp_servers/           # MCP 服务器示例
│   └── demo_server.py
│
└── reports/               # 生成的 Markdown 报告
```
