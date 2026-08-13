"""评测集 + 中间件效果评测 harness

- dataset/qa.jsonl      测试 query（按类别覆盖不同行为）
- micro_benchmarks.py   确定性微基准（不花 API，立即出百分比）
- run_eval.py           真实 E2E 评测（需 API key），支持 --ab 对比中间件开关
- metrics.py            指标计算与报告生成
"""
