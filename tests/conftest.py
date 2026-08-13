"""pytest 共享配置：把项目根目录加入 sys.path，方便直接 import 业务模块。

所有测试都不需要 API key —— 涉及 LLM / 网络的调用一律 mock 或走纯函数路径。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
