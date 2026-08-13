"""Sandbox —— 文件 / Shell 工具的受控执行环境。

安全设计
────────
1. 路径校验：`_safe_path` 用 `Path.is_relative_to()`（而不是字符串前缀），
   修复了 `workspace_evil` 兄弟目录被误判合法的问题。路径统一锚定到
   项目根目录下的 workspace/（不依赖运行时 cwd）。
2. bash 隔离：命令一律在 Docker 一次性容器内执行（`--network none`，
   仅挂载 workspace 目录）。Docker 不可用时返回明确错误，绝不回退原生 shell。
   超时通过 cidfile + `docker kill` 清理，防止残留容器。
"""

import os
import subprocess
import tempfile
from pathlib import Path

from config import CONFIG

# 项目根目录下的 workspace（绝对路径，避免 cwd 变化导致逃逸/找不到）
WORKSPACE = (Path(__file__).resolve().parent.parent / "workspace").resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)  # 目录不存在就创建


def _safe_path(path: str) -> Path:
    # 1. 拼到 workspace 后面
    full = WORKSPACE / path
    # 2. 转绝对路径（消除 .. 和符号链接）
    resolved = full.resolve()
    # 3. 检查是否真的在 workspace 内（组件边界比较，而非字符串前缀）
    if not resolved.is_relative_to(WORKSPACE):
        raise ValueError(f"路径越界: {path}")
    return resolved


def read_file(path: str) -> str:
    """Read a file from the workspace. Returns the file contents as a string."""
    p = _safe_path(path)

    if not p.exists():
        return f"文件不存在: {path}"

    return p.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace. Creates parent directories as needed."""
    p = _safe_path(path)

    p.parent.mkdir(parents=True, exist_ok=True)  # 自动创建父目录

    # 写文件
    p.write_text(content, encoding="utf-8")

    return f"写入成功: {path}"


def ls(dir_path: str = ".") -> str:
    """List files and directories in the workspace. Use '.' for the root workspace directory."""
    p = _safe_path(dir_path)

    if not p.exists():
        return f"目录不存在: {dir_path}"

    if not p.is_dir():
        return f"不是目录: {dir_path}"

    files = sorted(p.iterdir())

    if not files:
        return "(空目录)"
    return "\n".join(
        f"{'[DIR]' if f.is_dir() else '[FILE]'} {f.name}"
        for f in files
    )


def bash(command: str) -> str:
    """执行 shell 命令 —— 在 Docker 一次性容器内运行，实现真隔离。

    容器仅挂载 workspace，禁用网络。命令超时/输出过长都会被截断。
    Docker 不可用时返回错误，不回退宿主 shell。
    """
    cfg = CONFIG.get("sandbox", {}).get("bash", {})
    image = cfg.get("image", "python:3.12-slim")
    timeout = cfg.get("timeout", 10)
    max_output = cfg.get("max_output_chars", 5000)
    network = cfg.get("network", "none")
    memory = cfg.get("memory", "512m")
    cpus = cfg.get("cpus", 1)
    mount_point = cfg.get("mount_point", "/workspace")

    # ── 1. 检查 Docker 可用性（绝不回退原生 shell）──
    try:
        check = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=5
        )
        if check.returncode != 0:
            return "Docker 不可用或未启动。请先启动 Docker Desktop，再使用 bash 工具。"
    except FileNotFoundError:
        return "未找到 docker 命令。请安装并启动 Docker Desktop 后重试。"
    except subprocess.TimeoutExpired:
        return "Docker 检测超时。"

    # ── 2. 构造容器命令。cidfile 记录容器 id，超时时用于清理 ──
    fd, cid_path = tempfile.mkstemp(prefix="docker_cid_", dir=tempfile.gettempdir())
    os.close(fd)
    cmd = [
        "docker", "run", "--rm",
        "--network", network,
        "--memory", memory,
        "--cpus", str(cpus),
        "--cidfile", cid_path,
        "-v", f"{WORKSPACE}:{mount_point}",
        "-w", mount_point,
        image,
        "sh", "-c", command,
    ]

    try:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            # 清理可能残留的容器（--rm + docker kill）
            try:
                cid = Path(cid_path).read_text().strip()
                if cid:
                    subprocess.run(["docker", "kill", cid], capture_output=True, text=True)
            except Exception:
                pass
            return f"命令超时（{timeout} 秒）"
        except Exception as e:
            return f"命令执行出错: {e}"

        output = result.stdout or result.stderr or ""
        if result.returncode != 0 and result.stderr:
            output = f"(exit {result.returncode})\n{output}"
        if len(output) > max_output:
            output = output[:max_output] + f"\n...(截断，共 {len(output)} 字符)"
        return output or "(无输出)"

    finally:
        try:
            os.remove(cid_path)
        except OSError:
            pass
