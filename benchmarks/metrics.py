"""评测指标计算与报告生成。

Run schema（run_eval 每条 query 一个 run 的 dict）：
    {
        "id", "category", "question",
        "config": "with_middleware" | "baseline",
        "success": bool（是否产出 final_report）,
        "cost": float,
        "token_est": int（chars/2，与 TokenBudget 同口径）,
        "duration_s": float,
        "tool_call_counts": dict,
        "telemetry": dict（中间件埋点）,
        "error": str | None,
    }
"""

from __future__ import annotations


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def pct(a: float, b: float) -> float:
    """百分比，保留两位小数。"""
    return round(safe_div(a, b) * 100.0, 2)


def _telemetry(run: dict, key: str) -> dict:
    return run.get("telemetry", {}).get(key, {})


def aggregate_runs(runs: list[dict]) -> dict:
    """汇总一组 run（同一 config）的指标。"""
    n = len(runs) or 1
    success = [r for r in runs if r.get("success")]
    # 异常任务：要么被 ToolErrorHandling 捕获（telemetry.errors>0），要么直接崩 run（error 字段非空）
    err_runs = [
        r for r in runs
        if _telemetry(r, "tool_error").get("errors", 0) > 0 or r.get("error")
    ]
    err_success = [r for r in err_runs if r.get("success")]

    loop_t = [_telemetry(r, "loop_detection") for r in runs]
    tb_t = [_telemetry(r, "token_budget") for r in runs]
    sum_t = [_telemetry(r, "summarization") for r in runs]
    te_t = [_telemetry(r, "tool_error") for r in runs]

    tool_calls = [sum(r.get("tool_call_counts", {}).values()) for r in runs]
    sum_before = sum(t.get("chars_before", 0) for t in sum_t)
    sum_after = sum(t.get("chars_after", 0) for t in sum_t)

    return {
        "runs": len(runs),
        "report_success_rate": pct(len(success), len(runs)),
        "avg_token_est": round(safe_div(sum(r.get("token_est", 0) for r in runs), n), 1),
        "avg_cost": round(safe_div(sum(r.get("cost", 0.0) for r in runs), n), 6),
        "avg_tool_calls": round(safe_div(sum(tool_calls), n), 2),
        "loop": {
            "forced_stops": sum(t.get("forced_stops", 0) for t in loop_t),
            "repetition_events": sum(t.get("repetition_events", 0) for t in loop_t),
            "runs_with_forced_stop": sum(1 for t in loop_t if t.get("forced_stops", 0) > 0),
            "max_depth": max((t.get("max_depth", 0) for t in loop_t), default=0),
        },
        "token_budget": {
            "forced_stops": sum(t.get("forced_stops", 0) for t in tb_t),
            "max_chars": max((t.get("max_chars", 0) for t in tb_t), default=0),
        },
        "summarization": {
            "compressions": sum(t.get("compressions", 0) for t in sum_t),
            "auto_compacts": sum(t.get("auto_compacts", 0) for t in sum_t),
            "tightenings": sum(t.get("tightenings", 0) for t in sum_t),
            "chars_before": sum_before,
            "chars_after": sum_after,
            "compression_ratio": pct(sum_before - sum_after, sum_before),
            "last_utilization": max((t.get("utilization", 0.0) for t in sum_t), default=0.0),
        },
        "tool_error": {
            # 被捕获的异常 + 直接崩 run 的异常（基线没有中间件时以 error 字段体现）
            "errors": sum(t.get("errors", 0) for t in te_t)
            + sum(
                1 for r in runs
                if r.get("error") and not _telemetry(r, "tool_error").get("errors", 0)
            ),
            "runs_with_errors": len(err_runs),
            "recovery_rate": pct(len(err_success), len(err_runs)) if err_runs else None,
        },
    }


def compare_ab(with_runs: list[dict], base_runs: list[dict]) -> dict:
    """A/B 对比：中间件全开 vs 基线（只保留 run_guard 兜底）。"""
    a = aggregate_runs(with_runs)
    b = aggregate_runs(base_runs)

    def reduction(metric: str) -> float:
        base_val = b[metric]
        mw_val = a[metric]
        if base_val == 0:
            return 0.0
        return round((base_val - mw_val) / base_val * 100.0, 2)

    return {
        "tool_steps_reduction_pct": reduction("avg_tool_calls"),
        "token_reduction_pct": reduction("avg_token_est"),
        "cost_reduction_pct": reduction("avg_cost"),
        "report_success_rate_with": a["report_success_rate"],
        "report_success_rate_base": b["report_success_rate"],
        "loop_forced_stops": a["loop"]["forced_stops"],
        "loop_repetition_events": a["loop"]["repetition_events"],
        "tool_error_recovery_with": a["tool_error"]["recovery_rate"],
        "tool_error_recovery_base": b["tool_error"]["recovery_rate"],
        "errors_with": a["tool_error"]["errors"],
        "errors_base": b["tool_error"]["errors"],
    }


def render_markdown(
    per_config: dict[str, dict],
    ab: dict | None = None,
    title: str = "评测报告",
) -> str:
    """把聚合指标 + A/B 对比渲染成 markdown。"""
    lines = [f"# {title}", ""]

    for cfg, m in per_config.items():
        lines += [
            f"## 配置：{cfg}", "",
            f"- 任务数：{m['runs']}",
            f"- 报告成功率：{m['report_success_rate']}%",
            f"- 平均 token 估算：{m['avg_token_est']}",
            f"- 平均工具调用步数：{m['avg_tool_calls']}",
            f"- 平均成本：${m['avg_cost']:.6f}", "",
            "### LoopDetection",
            f"- 强制停止次数：{m['loop']['forced_stops']}（影响 {m['loop']['runs_with_forced_stop']} 个任务）",
            f"- 重复调用事件：{m['loop']['repetition_events']}，最大重复深度：{m['loop']['max_depth']}", "",
            "### TokenBudget / Summarization",
            f"- TokenBudget 强制停止：{m['token_budget']['forced_stops']}（预算 {m['token_budget']['max_chars']} tokens）",
            f"- Summarization：压缩 {m['summarization']['compressions']} 次（auto-compact {m['summarization']['auto_compacts']}，轻量截断 {m['summarization']['tightenings']} 次）",
            f"- 压缩率：{m['summarization']['compression_ratio']}%（{m['summarization']['chars_before']} → {m['summarization']['chars_after']} 字符）",
            f"- 上下文利用率峰值：{m['summarization']['last_utilization']}", "",
            "### ToolErrorHandling",
            f"- 工具异常数：{m['tool_error']['errors']}（{m['tool_error']['runs_with_errors']} 个任务出现异常）",
            f"- 异常恢复率：{m['tool_error']['recovery_rate']}%" if m['tool_error']['recovery_rate'] is not None else "- 异常恢复率：无异常样本",
            "",
        ]

    if ab:
        lines += [
            "## A/B 对比（中间件全开 vs 基线）", "",
            f"- 工具调用步数减少率：**{ab['tool_steps_reduction_pct']}%**",
            f"- Token 估算减少率：**{ab['token_reduction_pct']}%**",
            f"- 成本减少率：**{ab['cost_reduction_pct']}%**",
            f"- 报告成功率：中间件 {ab['report_success_rate_with']}% vs 基线 {ab['report_success_rate_base']}%",
            f"- 循环强制停止次数：{ab['loop_forced_stops']}（重复事件 {ab['loop_repetition_events']}）",
            f"- 工具异常恢复率：中间件 {ab['tool_error_recovery_with']}% vs 基线 {ab['tool_error_recovery_base']}%",
            f"- 工具异常数：中间件 {ab['errors_with']} vs 基线 {ab['errors_base']}",
            "",
        ]

    return "\n".join(lines)
