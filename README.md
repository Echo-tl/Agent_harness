# Agent Harness · Mini-DeerFlow-Agent

基于 **LangGraph** 的多智能体深度研究报告系统：输入一个问题，自动完成**任务拆解 → 多源检索 → 证据压缩 → 带引用报告的生成**，全程 SSE 实时流式输出。

> 定位：把"问一个问题 → 得到一份带可点击来源、可校验引用的 Markdown 研究报告"做成一个工程化的 Agent 系统——不只是跑通，还要**并发安全、有资源上限、可评测、可扩展**。

---

## ✨ 功能特性

- **多智能体协作**：`Planner`（一次性任务拆解）→ `Supervisor`（ReAct 总指挥，`task()` 委派）→ `Researcher` 子图（多源检索 + 压缩）→ `Report`（大纲→引言→章节→结论→参考资料）
- **多检索器并行**：Tavily 网页 / Arxiv 论文 / MCP 工具，`asyncio.gather` 并发；爬虫并发抓全文 + embedding 过滤
- **来源证据 + 引用校验**：每条发现绑定 `Evidence`（url/title/quote），报告自动生成"参考资料"节，内置校验器统计 **coverage / groundedness**，检测编造 URL
- **per-run 状态隔离**：`RunContext`（contextvar）按请求隔离 cost / visited_urls / evidence，并发请求互不污染；resume 时从 checkpoint **完整恢复**
- **真正生效的资源限制**：迭代次数 / 成本上限 / 总超时全部强制收尾（`RunGuard` + `asyncio.timeout`）
- **利用率驱动的渐进式上下文压缩**：`SummarizationMiddleware` 按"预估 token / 预算"算利用率——50% 起轻量截断旧工具结果、85% 触发 auto-compact（滑动窗口 + LLM 摘要）
- **全异步链路**：planner / compress / report 全部 `ainvoke`；embedding 请求自动分批（≤25 条/次），不阻塞事件循环
- **文件 / Shell 沙箱**：`workspace/` 路径组件级校验（`is_relative_to`）；`bash` 命令在 **Docker 一次性容器**内执行（禁网/限内存 CPU）
- **Skills 库（渐进式披露）**：上下文只放 `name: description` 索引，`use_skill()` 按需加载完整方法论（deep-research / 综述 / 论文审稿）
- **实时流式进度**：图在后台任务执行，工具执行中途的进度每 0.25s 实时推给前端，聊天式 UI 在左侧 agent 气泡里滚动显示
- **可评测**：`benchmarks/` 提供评测集 + 确定性微基准 + 真实 E2E A/B，量化中间件效果（循环抑制率 / Token 节省 / 工具异常恢复率）

---

## 🏗️ 架构

```
用户问题 ──→ planner ──→ supervisor ──→ [task()] ──→ researcher 子图 ──→ compress ──┐
                  │            (ReAct)         (search/rag/sandbox 工具)            │
                  └─────────────────────────────────────────────────────────────→ report → 最终报告
```

- **tools/** —— `search` / `rag_search` / `memory_search` / `use_skill` / sandbox 工具
- **retrievers/** —— `BaseRetriever`(+`asearch`) + tavily / arxiv / mcp
- **nodes/** —— planner / supervisor / compress / report / curator / filter
- **middleware/** —— Summarization / LoopDetection / TokenBudget / RunGuard / ToolError / DynamicContext / Clarification（全部可开关、带 telemetry 埋点）
- **runtime/** —— `RunContext`（per-run 状态 + 进度通道 + 利用率）
- **citations/** —— 引用抽取与校验（coverage / groundedness）
- **skills/** —— 渐进式披露的方法论库（3 个内置 skill + 模板）
- **benchmarks/** —— 评测集 + 微基准 + E2E runner

---

## 🚀 快速开始

### 1. 环境变量

复制 `.env.example` 为 `.env` 并填写（`.env` 已在 `.gitignore` 中，不会提交）：

| 变量 | 用途 |
|------|------|
| `DEEPSEEK_API_KEY` | 对话 LLM（deepseek-chat） |
| `DASHSCOPE_API_KEY` | embedding（text-embedding-v1） |
| `TAVILY_API_KEY` | 网页搜索 |

### 2. 安装

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
```

### 3. 运行

```bash
# 命令行模式
python main.py

# Web 服务（SSE 流式 + 聊天 UI）
python api.py
# 打开 http://localhost:8000
```

- `POST /research?question=...` → SSE 流，响应头 `X-Thread-ID`
- `POST /research/{thread_id}/resume?answer=...` → 回答 clarify 追问后继续

### 4. 测试 / 评测

```bash
python -m pytest tests/ -q                     # 单元测试（无需 API key）
python -m benchmarks.micro_benchmarks --out benchmarks/results/micro.md   # 确定性微基准（无需 API key）
python -m benchmarks.run_eval --dataset benchmarks/dataset/qa.jsonl --limit 2   # 真实 E2E（需 API key）
```

---

## ⚙️ 配置（config.py）

| 配置段 | 说明 |
|--------|------|
| `llm` / `embedding` | 模型、base_url、api_key、`llm.pricing`（成本单价）、`embedding.batch_size`（≤25 分批） |
| `search.retrievers` | 启用的检索器：`tavily` / `arxiv` / `mcp` |
| `rag` | 本地知识库目录、top_k |
| `filter` | 切块 / 相似度阈值 / top-N / 快速路径 |
| `summarization` | 利用率驱动的压缩参数（threshold / max_tokens / tighten / auto_compact） |
| `limits` | `max_researcher_tool_calls`、`max_supervisor_iterations`、`max_cost`、`timeout_seconds` |
| `sandbox.bash` | Docker 镜像 / 超时 / 网络 / 内存 / CPU 限制 |
| `skills` | skills 目录 + `enabled` 列表 |
| `middleware` | 中间件开关列表（评测 A/B 用） |

---

## 🛡️ 安全说明

- **文件沙箱**：`read_file`/`write_file`/`ls` 只允许 `workspace/`，路径用 `Path.is_relative_to()` 做组件边界校验，`workspace_evil` 兄弟目录、`../`、绝对路径、符号链接都会被拒绝。
- **bash 隔离（Docker）**：所有命令在一次性容器内执行，只挂载 workspace、默认禁网、限制内存/CPU、超时自动 `docker kill`。Docker 不可用时不回退宿主 shell。
- **密钥**：`.env` 不进版本库；报告与日志不落敏感信息。

> ⚠️ Windows + Docker Desktop：需在 Docker Desktop → Settings → Resources → File Sharing 中把项目所在盘符（如 `E:`）加入共享。

---

## 📚 Skills 库（渐进式披露）

`skills/<name>/SKILL.md`（frontmatter `name`/`description` + 方法论正文）+ 可选 `references/*.md` 模板。系统提示词只放轻量索引，模型判断任务匹配时调用 `use_skill('<name>')` 按需加载完整方法论。随附 3 个：

- `deep-research` —— 多角度搜索方法论、时间感知、质量校验清单
- `systematic-literature-review` —— 综述综合方法论（主题/共识/分歧/空白）+ APA/IEEE/BibTeX 引用模板
- `academic-paper-review` —— 单篇论文结构化审稿

添加新 skill：在 `skills/<name>/` 放 SKILL.md，加入 `config.skills.enabled` 即可，无需改代码。

---

## 🔎 引用与证据

- `search` 工具把真实爬取来源记录进 `RunContext.evidence`
- `compress` 让 LLM 从结果文本提取每条笔记的 `Evidence`（禁止编造 URL）
- `report` 末尾自动生成"参考资料"节，并用 `citations/verifier.py` 计算（只统计正文，不含自动生成的附录）：
  - **coverage**：正文句子带引用的比例
  - **groundedness**：引用 URL 落在证据集的比例（衡量是否编造）

---

## 📊 评测（量化中间件效果）

`benchmarks/` 量化 LoopDetection / TokenBudget / Summarization / ToolErrorHandling 的影响：

- **确定性微基准**（不花 API）：循环抑制率、Token 超预算拦截率、工具异常恢复率、压缩率
- **真实 E2E A/B**（需 API key）：`--ab` 对比"中间件全开 vs 基线"，输出循环步数减少率 / Token 减少率 / 成本减少率 / 报告成功率

指标口径见 `benchmarks/metrics.py`，原始数据输出到 `benchmarks/results/`（已 gitignore）。

---

## 📁 项目结构

```
├── api.py / main.py / streaming.py   # Web 服务 / CLI / SSE 流式
├── config.py                         # 全局配置
├── graph/__init__.py                 # LangGraph 图编排
├── nodes/                            # planner / supervisor / compress / report / curator / filter
├── middleware/                       # 可插拔中间件（带 telemetry 埋点）
├── runtime/context.py                # RunContext（per-run 状态 / 进度 / 利用率）
├── tools/  retrievers/  scraper/     # 工具 / 检索器 / 爬虫
├── sandbox/                          # 文件 / Docker bash 沙箱
├── citations/                        # 引用校验器
├── skills/                           # 渐进式披露的方法论库
├── benchmarks/                       # 评测集 + 微基准 + E2E runner
├── tests/                            # 单元测试（91 个，无需 API key）
└── knowledge/                        # 本地知识库（RAG 源）
```

---

## ⚠️ 已知限制

- Researcher 内嵌在 `task` 工具中同步等待（无真正并行子图）；后续可换 SubagentExecutor 异步执行
- MCP 需要 `langchain-mcp-adapters`（未安装时自动禁用）
- 知识库 `knowledge/*.txt` 需自行维护
- 真实 E2E 评测会消耗 API 费用

---

## 📄 License

MIT
