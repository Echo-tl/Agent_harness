"""Sandbox 路径校验与 Docker bash 隔离测试。

重点回归：`workspace_evil` 兄弟目录这类"字符串前缀"误判必须被拒绝。
"""

import subprocess
import pytest
from unittest import mock

from sandbox import tools
from sandbox.tools import WORKSPACE, _safe_path, read_file, write_file, ls, bash


# ── 路径校验（_safe_path）──

def test_safe_path_rejects_sibling_prefix():
    """workspace_evil 是 workspace 的兄弟目录，前缀相同但必须拒绝。"""
    with pytest.raises(ValueError):
        _safe_path("../workspace_evil")


def test_safe_path_rejects_parent_escape():
    with pytest.raises(ValueError):
        _safe_path("../../outside.txt")


def test_safe_path_rejects_absolute_windows_path():
    with pytest.raises(ValueError):
        _safe_path("C:/Windows/System32/drivers/etc/hosts")


def test_safe_path_rejects_absolute_unix_path():
    with pytest.raises(ValueError):
        _safe_path("/etc/passwd")


def test_safe_path_accepts_nested_within_workspace():
    p = _safe_path("a/b/c.txt")
    assert p.is_relative_to(WORKSPACE)


def test_safe_path_accepts_root():
    p = _safe_path(".")
    assert p == WORKSPACE


# ── 文件读写 ──

def test_write_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    write_file("dir/f.txt", "hello world")
    assert read_file("dir/f.txt") == "hello world"
    listing = ls("dir")
    assert "[FILE] f.txt" in listing


def test_read_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    assert "文件不存在" in read_file("nope.txt")


def test_write_blocks_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    with pytest.raises(ValueError):
        write_file("../../evil.txt", "x")


# ── bash：必须走 Docker，超时要 kill 容器，Docker 不可用要报错 ──

class _OkResult:
    returncode = 0
    stdout = "ok output"
    stderr = ""


def test_bash_runs_in_docker_container(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["kw"] = kw
        return _OkResult()

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = bash("echo hi")

    assert out == "ok output"
    # 必须是 docker run（绝不允许裸 shell）
    assert captured["cmd"][0] == "docker"
    assert captured["cmd"][1] == "run"
    # 挂载点 = workspace
    join = [a for a in captured["cmd"] if a == f"{tmp_path}:/workspace"]
    assert join, "workspace 必须挂载进容器"
    # shell 命令通过 sh -c 传进容器
    assert "sh" in captured["cmd"] and "-c" in captured["cmd"]


def test_bash_kills_container_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)

    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 10))

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = bash("sleep 999")
    assert "超时" in out


def test_bash_requires_docker_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)

    def fake_run(cmd, **kw):
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = bash("echo hi")
    assert "docker" in out.lower()  # 返回明确错误，而不是偷偷跑宿主 shell


def test_bash_reports_docker_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)

    class _Fail:
        returncode = 1
        stdout = ""
        stderr = "docker daemon not running"

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _Fail())
    out = bash("echo hi")
    assert "Docker 不可用" in out
