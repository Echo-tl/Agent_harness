"""
LLM 统一配置 —— 所有节点从这里导入，一处修改全局生效

用法：
    from llm_config import llm, llm_with_tools
"""

import os
from langchain_openai import ChatOpenAI

# ── 千问 API 配置（改这里一次，全项目生效）──
# LLM_CONFIG = {
#     "model": "qwen-plus",
#     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
#     "api_key": os.getenv("DASHSCOPE_API_KEY"),
#     "temperature": 0.0,
# }

from config import CONFIG

# pricing 是成本估算用的单价，不是 ChatOpenAI 的参数 —— 先取出来再实例化
LLM_CONFIG = dict(CONFIG["llm"])
LLM_PRICING = LLM_CONFIG.pop("pricing", {})

# ── 预创建实例（模块级单例，避免重复初始化）──
llm = ChatOpenAI(**LLM_CONFIG)

def add_cost(cost: float):
    """每次 LLM 调用后累加花费到"本次运行"的 RunContext。

    不再使用模块级变量：并发请求之间互不污染。
    """
    from runtime.context import get_run_context
    get_run_context().record_cost(cost)

def get_total_cost() -> float:
    """返回当前运行累计花费（美元）"""
    from runtime.context import get_run_context
    return get_run_context().cost

def reset_cost():
    """清零当前运行的成本计数。新研究开始时由 stream_research 重建 RunContext，
    这里仅作兼容兜底。"""
    from runtime.context import get_run_context
    get_run_context().cost = 0.0

def estimate_llm_cost(input_chars: int, output_chars: int) -> float:
    """根据输入输出字符数估算 LLM 调用花费（美元）。

    单价来自 config.py 的 CONFIG["llm"]["pricing"]（按实际配置的模型声明，
    默认 deepseek-chat），不再硬编码某个模型的价目表 —— 换模型后成本统计
    仍能对得上。
    粗略规则：中文 ~1.5 token/字，英文 ~0.25 token/字，折中取 0.6 token/字符。
    """
    input_tokens = input_chars * 0.6
    output_tokens = output_chars * 0.6
    cost = (
        (input_tokens / 1000) * LLM_PRICING.get("input_per_1k_tokens", 0.0)
        + (output_tokens / 1000) * LLM_PRICING.get("output_per_1k_tokens", 0.0)
    )
    return cost
