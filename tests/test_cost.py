"""LLM 成本估算测试 —— 单价来自 config 里声明的模型（deepseek-chat），不是硬编码。"""

import pytest

from config import CONFIG
from llm_config import estimate_llm_cost


def test_estimate_llm_cost_uses_configured_pricing():
    """1000 input + 1000 output chars，按 0.6 token/字 折算。

    deepseek-chat：input $0.27/M、output $1.10/M → 每 1K tokens 0.00027 / 0.0011。
    input  = 600 tokens * 0.00027 / 1K = 0.000162
    output = 600 tokens * 0.0011  / 1K = 0.00066
    """
    cost = estimate_llm_cost(1000, 1000)
    assert cost == pytest.approx(0.000162 + 0.00066, rel=1e-6)
    assert cost > 0


def test_pricing_declared_in_config():
    pricing = CONFIG["llm"]["pricing"]
    assert pricing["input_per_1k_tokens"] > 0
    assert pricing["output_per_1k_tokens"] > 0
    assert pricing["output_per_1k_tokens"] > pricing["input_per_1k_tokens"]


def test_zero_chars_cost_zero():
    assert estimate_llm_cost(0, 0) == 0.0
