"""Shared fixtures. No __init__.py here on purpose: upstream/ (on sys.path) has its own `tests`
package, and a second one would shadow it."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from td import config


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=15)
    except (subprocess.SubprocessError, OSError):
        return False
    return True


requires_docker = pytest.mark.skipif(not docker_available(), reason="docker daemon not running")


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    """Point every store and the ledger at a temp dir, so tests never touch real data or costs."""
    monkeypatch.setattr(config, "ROLLOUTS_DIR", tmp_path / "rollouts")
    monkeypatch.setattr(config, "JUDGEMENTS_DIR", tmp_path / "judgements")
    monkeypatch.setattr(config, "LEDGER_PATH", tmp_path / "costs" / "ledger.jsonl")
    monkeypatch.setattr(config, "LABELS_DIR", tmp_path / "labels")
    return tmp_path
