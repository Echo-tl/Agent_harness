"""真实 E2E 评测 runner —— 跑评测集，量化中间件效果。

需要 API key（DEEPSEEK / DASHSCOPE / TAVILY）。每条 query 会真实调用 LLM 与搜索，
耗时且消耗费用；建议先用 --limit 2 冒烟。

用法：
    python -m benchmarks.run_eval --dataset benchmarks/dataset/qa.jsonl \
        --out benchmarks/results [--ab] [--limit 2] [--concurrency 2]

- 默认只跑"完整中间件"配置。
- `--ab`：同一 query 再跑一次"基线"（只保留 run_guard 兜底），对比出各指标降低率。
- 输出 results.json + results.md。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import config as config_mod
from graph import build_parent_graph
from nodes.supervisor import reset_supervisor_agent
from runtime.context import RunContext, run_scope

from benchmarks.metrics import aggregate_runs, compare_ab, render_markdown

FULL_MW = [
    "summarization", "loop_detection", "tool_error", "dynamic_context",
    "token_budget", "run_guard", "clarification",
]
BASELINE_MW = ["run_guard"]  # A/B 基线：只留兜底，避免死循环烧钱


def _set_middleware(enabled: list[str]) -> None:
    config_mod.CONFIG["middleware"]["enabled"] = list(enabled)
    reset_supervisor_agent()  # Supervisor 惰性构建，需重建以读新配置


def _empty_run() -> dict:
    return {
        "success": False, "cost": 0.0, "token_est": 0, "duration_s": 0.0,
        "tool_call_counts": {}, "telemetry": {}, "error": None,
    }


async def _run_one(question: str, timeout: float) -> dict:
    """跑一次完整研究，返回 run 指标 dict。"""
    ctx = RunContext(thread_id=str(uuid.uuid4()))
    start = time.time()
    state = {}
    try:
        with run_scope(ctx):
            graph = build_parent_graph(checkpointer=InMemorySaver())
            async with asyncio.timeout(timeout):
                state = await graph.ainvoke(
                    {
                        "messages": [HumanMessage(content=question)],
                        "research_brief": "",
                        "final_report": "",
                        "total_cost": 0.0,
                        "citation_stats": {},
                    },
                    {"configurable": {"thread_id": ctx.thread_id}},
                )
    except TimeoutError:
        return _empty_run() | {"error": "timeout"}
    except Exception as e:
        return _empty_run() | {"error": f"{type(e).__name__}: {e}"}

    report = state.get("final_report") or ""
    msgs = state.get("messages", [])
    token_est = sum(len(str(m.content)) for m in msgs) // 2
    return {
        "success": bool(report.strip()),
        "cost": ctx.cost,
        "token_est": token_est,
        "duration_s": round(time.time() - start, 2),
        "tool_call_counts": dict(ctx.tool_call_counts),
        "telemetry": ctx.telemetry,
        "error": None,
    }


def _load_dataset(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


async def _evaluate(dataset: list[dict], ab: bool, limit: int | None, concurrency: int) -> list[dict]:
    timeout = float(config_mod.CONFIG["limits"].get("timeout_seconds", 600))
    if limit:
        dataset = dataset[:limit]
    sem = asyncio.Semaphore(max(1, concurrency))
    runs: list[dict] = []

    async def _run_with_config(item: dict, cfg: str) -> dict:
        _set_middleware(FULL_MW if cfg == "with_middleware" else BASELINE_MW)
        async with sem:
            r = await _run_one(item["question"], timeout)
        r.update(
            id=item["id"], category=item["category"], question=item["question"], config=cfg
        )
        print(
            f"[eval] {item['id']} [{cfg}] success={r['success']} "
            f"cost=${r['cost']:.4f} tok={r['token_est']} dur={r['duration_s']}s {r.get('error') or ''}"
        )
        return r

    for item in dataset:
        runs.append(await _run_with_config(item, "with_middleware"))
        if ab:
            runs.append(await _run_with_config(item, "baseline"))
    return runs


async def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="真实 E2E 评测（需 API key）")
    parser.add_argument("--dataset", type=str, default="benchmarks/dataset/qa.jsonl")
    parser.add_argument("--out", type=str, default="benchmarks/results")
    parser.add_argument("--ab", action="store_true", help="额外跑基线并做 A/B 对比")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args(argv)

    dataset = _load_dataset(Path(args.dataset))
    print(f"[eval] 评测集 {len(dataset)} 条，limit={args.limit}, ab={args.ab}")

    runs = await _evaluate(dataset, args.ab, args.limit, args.concurrency)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    per_config = {}
    for cfg in {r["config"] for r in runs}:
        per_config[cfg] = aggregate_runs([r for r in runs if r["config"] == cfg])

    ab = None
    if args.ab:
        with_runs = [r for r in runs if r["config"] == "with_middleware"]
        base_runs = [r for r in runs if r["config"] == "baseline"]
        ab = compare_ab(with_runs, base_runs)

    report = render_markdown(per_config, ab=ab, title="真实 E2E 评测报告")
    (out_dir / "results.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"[eval] 已写入 {out_dir / 'results.json'} 与 {out_dir / 'results.md'}")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_main(argv))


if __name__ == "__main__":
    main()
