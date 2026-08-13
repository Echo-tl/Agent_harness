"""全局配置 —— 一处修改，全项目生效"""

from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量

import os

CONFIG = {
    # ── LLM 聊天模型 ──
    "llm": {
        "model" : "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "temperature": 0.0,
        # 计费单价（美元 / 1K tokens），供 llm_config.estimate_llm_cost 估算花费。
        # 换成其它模型时记得同步改这里，否则成本统计会按旧模型算。
        # deepseek-chat 参考价（2025+）：输入 $0.27/M、输出 $1.10/M。
        "pricing": {
            "input_per_1k_tokens": 0.00027,
            "output_per_1k_tokens": 0.0011,
        },
    },

    # ── Embedding 向量化模型 ──
    "embedding": {
        "model": "text-embedding-v1",
        "base_url": "https://ws-d3thnk1of6bc4zr8.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "batch_size": 25,   # 单次请求最多条数（百炼 API 上限），超过会自动分批
    },

    # ── Web 搜索
    "search": {
        "max_results": 5,
        "api_key": os.getenv("TAVILY_API_KEY"),
        "retrievers": ["tavily", "arxiv", "mcp"],
    },

    # ── RAG 本地知识库 ──
    "rag": {
        "knowledge_dir": "knowledge",
        "top_k": 3,          # 每次检索返回几条
    },

    "mcp": {
        "servers": [
            {
                "name": "demo",
                "command": "python",
                "args": ["mcp_servers/demo_server.py"],
            }
            # {
            #     "name": "mcp_server_1",
            #     "command": "python",
            #     "args": ["my_mcp_server.py"],
            # },
            # {
            #     "name": "mcp_server_2",
            #     "connection_url": "wss://mcp.example.com/ws",,
            #     "connection_token": "your_token",
            # },
        ]
    },

    # 爬虫/数据抓取模块的配置参数
    "scraper": {
        "engine": "bs",   # 指定使用 BeautifulSoup4（简称 bs4）作为 HTML 解析库
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",  # 告诉目标网站“你是谁”。这里伪装成一台 Windows 系统上的 Chrome 浏览器
        "timeout": 15,  # 发起 HTTP 请求后，15 秒内没有得到网站的响应，程序就会主动放弃
        "max_concurrency": 3,   # 同时最多允许 3 个网页抓取任务并行执行
        "min_content_length": 100,  # 如果抓取到的网页正文长度小于 100 个字符，程序会认为这是一个无效页面
    },

    # Embedding 过滤配置
    "filter": {
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "similarity_threshold": 0.35, # 相似度阈值
        "max_chunks": 10,  # 最多只取排名最靠前的 10 个文本块喂给大模型
        "fast_path_max_chars": 8000, # 少于 8000 字符，直接输出
    },

    # Summarization 上下文压缩（利用率驱动的渐进式压缩）
    "summarization": {
        "threshold_chars": 3000,            # 绝对阈值：预估 token 超此触发动态摘要
        "max_tokens": 10000,               # 利用率分母（上下文预算；按 agent 覆盖）
        "tighten_utilization": 0.50,        # 利用率 ≥ 此值：轻量级截断旧工具结果（收紧体积）
        "auto_compact_utilization": 0.85,   # 利用率 ≥ 此值：全量 LLM 摘要（auto-compact）
    },

    # Report 输出格式
    "report": {
        "report_format": "apa",
        "total_words": 1500,
        "tone": "objective",
        "language": "chinese",
    },

    # Temperature 分层策略
    "temperature": {
        "tool_selection": 0.0,
        "curator": 0.2,
        "compress": 0.25,
        "report": 0.35,
        "introduction": 0.25,
        "conclusion": 0.25,
        "reflection": 0.0,
    },

    # ── 迭代/成本/超时上限（全项目统一，已真正参与路由）──
    "limits": {
        "max_researcher_tool_calls": 2,    # Researcher 子图内最多搜几轮
        "max_supervisor_iterations": 3,    # Supervisor 最多调用几次 task
        "max_cost": 1.0,                   # 单次研究累计成本上限（美元），超过强制收尾
        "timeout_seconds": 600,            # 整次研究最长执行时间（秒），超过发超时事件
    },

    # ── Sandbox 文件/Shell 工具 ──
    "sandbox": {
        "workspace_dir": "workspace",
        "bash": {
            "mode": "docker",        # bash 一律在 Docker 一次性容器内执行（真隔离）
            "image": "python:3.12-slim",
            "timeout": 10,           # 单条命令超时（秒）
            "max_output_chars": 5000,# 输出截断长度
            "network": "none",       # 容器网络："none"（默认，禁用）或 "bridge"
            "memory": "512m",        # 容器内存上限
            "cpus": 1,               # 容器 CPU 上限
            "mount_point": "/workspace",  # 工作区挂载到容器内的路径
        },
    },

    # ── Skills 库（静态加载，不做按问题动态筛选）──
    # 启动时把 enabled 里的 skill 方法论全部拼进 agent 上下文。
    # 往 skills/<name>/ 放一个 SKILL.md 并加入列表即可扩展；清空列表=全部加载。
    "skills": {
        "dir": "skills",
        "enabled": [
            "deep-research",
            "systematic-literature-review",
            "academic-paper-review",
        ],
    },

    # ── 中间件开关（A/B 评测用）──
    # Researcher / Supervisor 只挂载列表里的中间件；清空列表=全部挂载。
    # 评测基线时改成 ["run_guard"]（只保留兜底），对比完整配置即可量化各中间件效果。
    "middleware": {
        "enabled": [
            "summarization",
            "loop_detection",
            "tool_error",
            "dynamic_context",
            "token_budget",
            "run_guard",
            "clarification",
        ],
    },
}