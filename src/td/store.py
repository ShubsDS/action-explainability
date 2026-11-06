"""Append-only rollout and judgement stores, with deterministic cache keys (handoff rule 7, §4).

A paid sample is written the moment it arrives (fsync'd) and never regenerated if its key is
already on disk. Records are one JSON object per line. A torn final line (crash mid-write) is
skipped with a warning on load; its sample is simply re-requested.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from td import config

log = logging.getLogger(__name__)

# Minimum fields of a rollout row (handoff §4). `append` refuses a row missing any of them.
ROLLOUT_FIELDS = (
    "rollout_id", "checkpoint", "task_id", "split", "heldout", "condition",
    "parent_rollout_id", "fork_token", "insert_id",
    "prompt_tokens", "output_tokens",
    "reasoning", "answer_text", "code",
    "truncated", "n_prompt_tokens", "n_output_tokens",
    "grade", "accidental_pass", "cls",
    "sampler", "upstream_sha", "created_at", "cost_usd",
)

JUDGEMENT_FIELDS = ("rollout_id", "judge", "prompt_version", "model", "output", "created_at", "cost_usd")


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(*, checkpoint: dict, task_id: str, split: str, condition: str = "base",
              parent_rollout_id: str | None = None, fork_token: int | None = None,
              insert_id: str | None = None, sample_idx: int, sampler: dict) -> dict:
    """Everything that makes a sample distinct. No sampler seed: Tinker exposes none (handoff §3)."""
    ck = {k: checkpoint[k] for k in ("character", "seed", "step", "sampler_path")}
    sm = {k: sampler[k] for k in ("temperature", "max_tokens")}
    return {"checkpoint": ck, "task_id": task_id, "split": split, "condition": condition,
            "parent_rollout_id": parent_rollout_id, "fork_token": fork_token, "insert_id": insert_id,
            "sample_idx": sample_idx, "sampler": sm}


def rollout_id(key: dict) -> str:
    return hashlib.sha256(_canon(key).encode()).hexdigest()[:32]


def judgement_key(rollout_id: str, judge: str, prompt_version: str, model: str) -> str:
    return hashlib.sha256(_canon([rollout_id, judge, prompt_version, model]).encode()).hexdigest()[:32]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _JsonlStore:
    """Append-only JSONL file with an in-memory index on `id_field`."""

    id_field: str

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index: dict[str, dict] = {}
        self._load()

    def _row_id(self, row: dict) -> str:
        return row[self.id_field]

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            data = f.read()
        bad = 0
        for i, line in enumerate(data.splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                log.warning("%s: skipping unparseable line %d (torn write?)", self.path, i + 1)
                continue
            self._index[self._row_id(row)] = row
        if data and not data.endswith(b"\n"):
            with open(self.path, "ab") as f:  # isolate a torn tail so the next append starts clean
                f.write(b"\n")
        if bad:
            log.warning("%s: %d unparseable line(s) skipped", self.path, bad)

    def __contains__(self, rid: str) -> bool:
        return rid in self._index

    def __len__(self) -> int:
        return len(self._index)

    def get(self, rid: str) -> dict | None:
        return self._index.get(rid)

    def rows(self):
        return iter(self._index.values())

    def _append(self, row: dict) -> None:
        rid = self._row_id(row)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            if rid in self._index:
                raise FileExistsError(f"{self.path}: {self.id_field}={rid} already stored (append-only)")
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            self._index[rid] = row


class RolloutStore(_JsonlStore):
    id_field = "rollout_id"

    def __init__(self, name: str, root: Path | None = None):
        super().__init__((root or config.ROLLOUTS_DIR) / f"{name}.jsonl")

    def append(self, row: dict) -> None:
        missing = [k for k in ROLLOUT_FIELDS if k not in row]
        if missing:
            raise ValueError(f"rollout row missing fields: {missing}")
        self._append(row)


class JudgementStore(_JsonlStore):
    """Keyed by (rollout_id, judge, prompt_version, model). A new prompt version is a new row;
    nothing is ever overwritten."""
    id_field = "judgement_id"

    def __init__(self, name: str, root: Path | None = None):
        super().__init__((root or config.JUDGEMENTS_DIR) / f"{name}.jsonl")

    def append(self, row: dict) -> None:
        missing = [k for k in JUDGEMENT_FIELDS if k not in row]
        if missing:
            raise ValueError(f"judgement row missing fields: {missing}")
        row = {"judgement_id": judgement_key(row["rollout_id"], row["judge"], row["prompt_version"],
                                             row["model"]), **row}
        self._append(row)

    def has(self, rollout_id: str, judge: str, prompt_version: str, model: str) -> bool:
        return judgement_key(rollout_id, judge, prompt_version, model) in self
