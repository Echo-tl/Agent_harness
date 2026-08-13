"""确定性微基准 —— 不调用任何外部 API，直接驱动中间件类，立即产出关键百分比。

每个基准都在 run_scope 里跑，让中间件的 telemetry 埋点真实落进 RunContext，
再据此算指标。模拟中的"预算上限"等参数是脚本假设，报告里已标注。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from middleware.loop_detection import LoopDetectionMiddleware
from middleware.token_budget import TokenBudgetMiddleware
from middleware.summarization import SummarizationMiddleware
from middleware.tool_error_handling import ToolErrorHandlingMiddleware
from runtime.context import RunContext, run_scope


# ── 工具 ──
def _tool_msg(name: str = "search", args: dict | None = None) -> AIMessage:
    m = AIMessage(content="")
    m.tool_calls = [{"name": name, "args": args or {"query": "same"}, "id": "c1"}]
    return m


class _FakeSummary:
    content = "这是压缩后的摘要。"


# ── 1. LoopDetection：无效循环抑制 ──
def loop_benchmark(budget_steps: int = 20, repeat_before_stop: int = 5) -> dict:
    """模拟模型反复调用同一工具。

    有 LoopDetection：第 repeat_before_stop 次重复时强制停止。
    无中间件：循环会一直重复，直到 RunGuard/TokenBudget 在 budget_steps 步兜底停止。
    """
    mid = LoopDetectionMiddleware(name="loop_eval")
    ctx = RunContext(thread_id="loop_eval")
    steps_with = budget_steps
    with run_scope(ctx):
        for i in range(1, budget_steps + 1):
            steps_with = i
            out = mid.after_model({"messages": [HumanMessage("hi"), _tool_msg()]}, None)
            if out is not None:
                break
    t = ctx.telemetry.get("loop_detection", {})
    return {
        "steps_with_middleware": steps_with,
        "steps_without_middleware": budget_steps,  # 模拟：无中间件时由其它兜底在第 20 步停
        "loop_steps_reduction_pct": round(
            (budget_steps - steps_with) / budget_steps * 100, 2
        ),
        "forced_stops": t.get("forced_stops", 0),
        "repetition_events": t.get("repetition_events", 0),
    }


# ── 2. TokenBudget：超预算拦截 ──
def token_benchmark(
    max_tokens: int = 10000,
    base_chars: int = 2000,
    step_chars: int = 2000,
    context_cap_tokens: int = 50000,
) -> dict:
    """模拟上下文逐轮增长（每轮 +step_chars 字符）。

    有 TokenBudget：超过 max_tokens 即强制停止。
    无中间件：会一直累积到模拟的上下文上限 context_cap_tokens。
    """
    mid = TokenBudgetMiddleware(name="token_eval", max_tokens=max_tokens)
    ctx = RunContext(thread_id="token_eval")
    steps = 0
    tokens_at_stop = 0
    with run_scope(ctx):
        for i in range(1, 100):
            steps = i
            content_chars = base_chars + i * step_chars
            # 最后一条必须有 tool_calls，否则中间件只是置 stopped 标记、不返回停止动作
            messages = [HumanMessage(content="x" * content_chars), _tool_msg()]
            tokens_at_stop = sum(len(str(m.content)) for m in messages) // 2
            out = mid.after_model({"messages": messages}, None)
            if out is not None:
                break
    return {
        "steps_until_stop": steps,
        "tokens_at_stop": tokens_at_stop,
        "max_tokens": max_tokens,
        "over_budget_tokens": max(0, tokens_at_stop - max_tokens),
        "context_cap_tokens": context_cap_tokens,  # 模拟：无中间件时的上下文上限
        "tokens_saved_pct": round(
            max(0, context_cap_tokens - tokens_at_stop) / context_cap_tokens * 100, 2
        ),
        "forced_stops": ctx.telemetry.get("token_budget", {}).get("forced_stops", 0),
    }


# ── 3. ToolErrorHandling：工具异常恢复 ──
class _FakeRequest:
    def __init__(self, tid: str):
        self.tool_call = {"id": tid}


def tool_error_benchmark(total: int = 10, failing: int = 4) -> dict:
    """模拟 total 次工具调用，其中 failing 次抛异常。

    有 ToolErrorHandling：异常全部捕获转为 ToolMessage，run 继续（恢复）。
    无中间件：异常向上抛，直接崩 run。
    """
    mid = ToolErrorHandlingMiddleware()
    ctx = RunContext(thread_id="tool_error_eval")

    def handler(req, i: int):
        if i < failing:
            raise RuntimeError(f"tool {i} failed")
        return "ok"

    recovered = 0
    with run_scope(ctx):
        for i in range(total):
            result = mid.wrap_tool_call(_FakeRequest(f"c{i}"), lambda r, i=i: handler(r, i))
            if hasattr(result, "content") and "工具执行出错" in result.content:
                recovered += 1

    crashed_without = 0
    for i in range(failing):
        try:
            handler(_FakeRequest(f"c{i}"), i)
        except RuntimeError:
            crashed_without += 1

    return {
        "calls": total,
        "failing_calls": failing,
        "errors_caught": ctx.telemetry.get("tool_error", {}).get("errors", 0),
        "recovered": recovered,
        "recovery_rate_with_pct": round(recovered / failing * 100, 2),
        "crashed_without_middleware": crashed_without,
    }


# ── 4. Summarization：上下文压缩率 ──
def summarization_benchmark() -> dict:
    """模拟超过阈值的会话；假 LLM 返回短摘要。测压缩前后字符。"""
    import middleware.summarization as summ_mod

    class _FakeLLM:
        def invoke(self, prompt):
            return _FakeSummary()

    old_llm = summ_mod.llm
    summ_mod.llm = _FakeLLM()
    try:
        mid = SummarizationMiddleware(name="summ_eval", threshold_chars=10)
        messages = [HumanMessage(content="这是需要压缩的很长的上下文内容，" * 200) for _ in range(10)]
        ctx = RunContext(thread_id="summ_eval")
        with run_scope(ctx):
            result = mid.before_model({"messages": messages}, None)
    finally:
        summ_mod.llm = old_llm

    before = sum(len(m.content) for m in messages)
    after = sum(len(m.content) for m in result["messages"]) if result else before
    ratio = (before - after) / before if before else 0
    return {
        "compressions": ctx.telemetry.get("summarization", {}).get("compressions", 0),
        "chars_before": before,
        "chars_after": after,
        "compression_ratio_pct": round(ratio * 100, 2),
    }


# ── 汇总报告 ──
def run_micro_benchmarks() -> str:
    loop = loop_benchmark()
    token = token_benchmark()
    tool = tool_error_benchmark()
    summ = summarization_benchmark()

    return f"""# 中间件微基准（确定性模拟，不调用 API）

## 1. LoopDetection —— 无效循环抑制率
- 有中间件：在第 **{loop['steps_with_middleware']}** 次重复调用时强制停止（telemetry: forced_stops={loop['forced_stops']}, repetition_events={loop['repetition_events']}）
- 无中间件（模拟）：由其它兜底在第 **{loop['steps_without_middleware']}** 步才停
- **无效循环步数减少率：{loop['loop_steps_reduction_pct']}%**

## 2. TokenBudget —— 超预算拦截
- 有中间件：在第 **{token['steps_until_stop']}** 轮、token 估算 **{token['tokens_at_stop']}** 时强制停止（预算 {token['max_tokens']}，超出 {token['over_budget_tokens']}）
- 无中间件（模拟）：会累积到上下文上限 **{token['context_cap_tokens']}**
- **Token 消耗节省率（模拟）：{token['tokens_saved_pct']}%**；强制停止次数：{token['forced_stops']}

## 3. ToolErrorHandling —— 工具异常恢复率
- 有中间件：{tool['calls']} 次调用中 {tool['failing_calls']} 次抛异常，**全部捕获为 ToolMessage（恢复 {tool['recovered']}/{tool['failing_calls']}）**
- 无中间件：{tool['crashed_without_middleware']} 次异常直接崩溃
- **异常恢复率：有中间件 {tool['recovery_rate_with_pct']}% vs 无中间件 0%**

## 4. Summarization —— 上下文压缩率
- 压缩次数：{summ['compressions']}
- 字符 {summ['chars_before']} → {summ['chars_after']}
- **压缩率：{summ['compression_ratio_pct']}%**

> 注：微基准是确定性的组件级模拟（预算/上限为脚本假设），用于快速对齐口径；
> 端到端真实数字请运行 `python -m benchmarks.run_eval`。
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="中间件确定性微基准（无 API）")
    parser.add_argument("--out", type=str, default=None, help="输出 markdown 文件路径")
    args = parser.parse_args(argv)

    report = run_micro_benchmarks()
    print(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"[micro_benchmarks] 已写入 {out}")


if __name__ == "__main__":
    main()
