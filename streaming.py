"""流式输出模块 —— 把 graph 的执行过程实时推给前端"""

import json
import asyncio
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from graph import build_parent_graph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from runtime.context import RunContext, run_scope, run_context_to_dict
from config import CONFIG

# ── 事件类型定义 ──
# 每个节点执行完，根据 node_name 产出一个 dict
# "type" 字段前端用来决定怎么渲染
# update 通常是当前节点接收的"完整状态"字典，可以理解为它代表"全局图状态"在当前时刻的快照

# ── SSE 事件统一结构 ──
# 所有事件统一为：{type, message, data}
#   - type:    事件类型（plan / dispatch / complete / report / clarify / error）
#   - message: 一行人类可读的描述（所有事件都有）
#   - data:    该类型专属的结构化载荷
# 为了兼容前端（index.html）直接读 event.research_brief / event.topics /
# event.content / event.question / event.status 等旧平铺字段，data 的内容会
# 再平铺一份到事件顶层。前端无需改动。
def _event(type_: str, message: str, **data) -> dict:
    ev = {"type": type_, "message": message, "data": data}
    ev.update(data)  # 平铺镜像：data 的字段同时挂在顶层
    return ev


def _format_event(namespace, node_name, update):
    """根据 update 内容推断事件类型，不依赖 node_name"""

    # 保护：update 必须是 dict
    if not isinstance(update, dict):
        return None

    # ── 事件1: 研究计划生成（有 research_brief）──
    if "research_brief" in update and update["research_brief"]:
        return _event(
            "plan",
            "研究计划已生成",
            research_brief=update["research_brief"],
        )

    # ── 事件2: 最终报告生成（附带引用校验统计）──
    if "final_report" in update and update["final_report"]:
        data = {"content": update["final_report"]}
        if update.get("citation_stats"):
            data["citation_stats"] = update["citation_stats"]
        return _event("report", "最终报告已生成", **data)

    # ── 事件3: Supervisor 派发研究（messages 里最后一条 AIMessage 调用了 task / ResearchComplete）──
    msgs = update.get("messages", [])
    if msgs and hasattr(msgs[-1], "tool_calls") and msgs[-1].tool_calls:
        last = msgs[-1]
        tool_names = [tc.get("name") for tc in last.tool_calls]
        if "task" in tool_names:
            topics = [
                tc.get("args", {}).get("research_topic", "")[:100]
                for tc in last.tool_calls
                if tc.get("name") == "task"
            ]
            return _event(
                "dispatch",
                f"正在研究 {len(topics)} 个主题",
                topics=topics,
            )
        if "ResearchComplete" in tool_names:
            return _event("complete", "研究已收集足够信息")

    # ── 其余内部节点不推事件 ──
    return None

# ── SSE 格式化 ──
def _sse(event: dict) -> str:
    """把一个事件 dict 转成 SSE 格式的字符串"""

    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ── 从 astream 的 chunk 中提取 (namespace, node_name, update) ──
# astream(subgraphs=True) 的返回格式取决于 LangGraph 版本
# 可能是 chunk = {node_name: update} （父图）
# 或 chunk = {(namespace, node_name): update}（子图）

def _parse_chunk(chunk):
    """把一个 astream chunk 拆成 [(namespace, node_name, update), ...]"""

    events = []
    # chunk = ((namespace, node_name), update_dict)
    key, update = chunk
    # 情况1: key = ("subgraph:...", "node_name")  → 子图
    if isinstance(key, tuple) and len(key) > 2:
        namespace = key[:-1]   # 去掉最后一个 = node_name
        node_name = key[-1]    # 最后一个 = node_name

    # 情况2: key = ()  → 父图，node_name 在 update 的 key 里
    elif isinstance(key, tuple) and len(key) == 0:
        namespace = ()
        node_name = list(update.keys())[0]
        update = update[node_name]   # 剥掉外层，拿到真正的更新数据

    # 情况3: key = "planner"  → 老版本父图（字符串）
    else:
        namespace = ()
        node_name = key
    events.append((namespace, node_name, update))

    return events


def _timeout_seconds() -> float:
    return float(CONFIG["limits"].get("timeout_seconds", 600))


# 外层节点 → 进度文案（Researcher 子图内部由工具 push_progress 上报细节）
_NODE_PROGRESS = {
    "planner": "正在制定研究计划…",
    "supervisor": "正在规划并派发研究任务…",
    "compress": "正在压缩研究结果…",
    "final_report": "正在撰写最终报告…",
}


def _drain_progress(ctx) -> list[str]:
    """取出并清空本轮节点/工具 push 的进度消息（SSE progress 事件的来源）。"""
    if not ctx.progress_events:
        return []
    msgs = list(ctx.progress_events)
    ctx.progress_events.clear()
    return msgs


async def _yield_stream(ctx, stream, timeout: float) -> AsyncGenerator[str, None]:
    """把 graph.astream 的 chunk 转成 SSE 事件，并在 chunk 间隙实时排空进度。

    图在后台任务里跑，工具执行中途 push 的进度消息会被这里每 0.25s 轮询一次
    实时转发给前端（而不是等整个 super-step 跑完才发）。chunk 顺序不变。
    """
    queue: asyncio.Queue = asyncio.Queue()
    _last: list[str | None] = [None]

    async def _run():
        try:
            async for chunk in stream:
                await queue.put(("chunk", chunk))
            await queue.put(("end", None))
        except Exception as e:
            await queue.put(("error", e))

    task = asyncio.create_task(_run())
    try:
        async with asyncio.timeout(timeout):
            while True:
                # 实时排空进度（工具执行中途 push 的也能立刻发出）
                for msg in _drain_progress(ctx):
                    if msg != _last[0]:
                        _last[0] = msg
                        yield _sse(_event("progress", msg))

                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue  # 没有新 chunk，先回去再看有没有新进度
                if kind == "end":
                    break
                if kind == "error":
                    yield _sse(_event("error", f"研究失败: {payload}"))
                    return

                for namespace, node_name, update in _parse_chunk(payload):
                    label = _NODE_PROGRESS.get(node_name)
                    if label and label != _last[0]:
                        _last[0] = label
                        yield _sse(_event("progress", label))
                    event = _format_event(namespace, node_name, update)
                    if event is not None:
                        yield _sse(event)
    except TimeoutError:
        yield _sse(_event("error", f"研究超时（超过 {timeout} 秒），已停止。"))
    except Exception as e:
        yield _sse(_event("error", f"研究失败: {e}"))
    finally:
        if not task.done():
            task.cancel()


# ── 主函数: 流式执行研究 ──
async def stream_research(question: str, thread_id: str) -> AsyncGenerator[str, None]:
    """
      发起研究，流式返回 SSE 事件。
      如果遇到 interrupt，yield 一个 clarify 事件后结束。
      调用方需要检查最后一个事件看是否需要用户回复。

      每次请求创建独立的 RunContext（thread_id），整段图执行受
      timeout_seconds 限制；超时/异常时发出 error 事件而非崩溃。
    """
    ctx = RunContext(thread_id=thread_id)

    with run_scope(ctx):
        async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
            graph = build_parent_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": thread_id}}

            initial_state = {
                "messages": [HumanMessage(content=question)],
                "research_brief": "",
                "final_report": "",
                "total_cost": 0.0,
                "citation_stats": {},
            }

            # 2. 流式执行（受总超时限制，进度实时转发）
            async for sse in _yield_stream(ctx, graph.astream(initial_state, config, subgraphs=True), _timeout_seconds()):
                yield sse

            # 3. 检查是否被 interrupt
            state = await graph.aget_state(config)
            if state.next:
                # 有 interrupt：先把 RunContext 快照写进 checkpoint，
                # 下一次 resume 请求据此完整恢复（cost/visited_urls/evidence/计数）
                try:
                    await graph.aupdate_state(config, {"run_ctx_data": run_context_to_dict(ctx)})
                except Exception as e:
                    print(f"[Stream] 持久化 RunContext 失败: {e}")

                # 有 interrupt，取出问题
                question_text = ""
                if state.tasks:
                    for task in state.tasks:
                        if task.interrupts:
                            question_text = task.interrupts[0].value
                            break

                yield _sse(_event("clarify", "需要补充信息", status="waiting", question=question_text))

# ── 恢复函数: 用户回复后继续 ──
async def stream_resume(thread_id: str, answer: str) -> AsyncGenerator[str, None]:
    """用户回答了 clarify 的追问，继续执行。

    为同一个 thread_id 重新建立 RunContext，并从 checkpoint 里保存的
    run_ctx_data 快照完整恢复（cost / visited_urls / evidence / 计数 /
    middleware 状态），保证同一对话跨请求状态连续。
    """
    ctx = RunContext(thread_id=thread_id)

    with run_scope(ctx):
        async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
            graph = build_parent_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": thread_id}}

            # 从 checkpoint 完整恢复 RunContext
            try:
                prev = await graph.aget_state(config)
                if prev.values:
                    snapshot = prev.values.get("run_ctx_data")
                    if snapshot:
                        ctx.restore_from(snapshot)
                    else:
                        # 旧 checkpoint 没有快照：至少恢复累计成本
                        ctx.cost = float(prev.values.get("total_cost", 0.0))
            except Exception:
                pass

            # 继续执行（受总超时限制，进度实时转发）
            async for sse in _yield_stream(
                ctx,
                graph.astream(Command(resume=answer), config, subgraphs=True, stream_mode="updates"),
                _timeout_seconds(),
            ):
                yield sse

            # 再次检查 interrupt（可能有多轮追问）
            state = await graph.aget_state(config)
            if state.next:
                # 再次中断 → 持久化更新后的 RunContext 快照，供下一轮 resume 恢复
                try:
                    await graph.aupdate_state(config, {"run_ctx_data": run_context_to_dict(ctx)})
                except Exception as e:
                    print(f"[Stream] 持久化 RunContext 失败: {e}")

                # 有 interrupt，取出问题
                question_text = ""
                if state.tasks:
                    for task in state.tasks:
                        if task.interrupts:
                            question_text = task.interrupts[0].value
                            break

                yield _sse(_event("clarify", "需要补充信息", status="waiting", question=question_text))
