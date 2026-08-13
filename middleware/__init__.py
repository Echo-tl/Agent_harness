"""Middleware 包 —— 可插拔行为注入"""

from config import CONFIG


def enabled_middleware(instances: list) -> list:
    """按 CONFIG["middleware"]["enabled"] 过滤中间件实例。

    每个中间件类声明 `key`（类型名）；enabled 未配置或为空时全部挂载。
    评测（benchmarks/run_eval.py --ab）用它做 A/B 基线。
    """
    enabled = CONFIG.get("middleware", {}).get("enabled")
    if not enabled:
        return instances
    return [
        m for m in instances
        if getattr(m, "key", type(m).__name__) in enabled
    ]
