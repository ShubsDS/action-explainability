"""Impossible-LiveCodeBench tasks, loaded through upstream's `load_impossible_lcb`.

The held-out split is upstream's sha256-of-task-id split (22 held-out / 81 training tasks at
heldout_frac=0.25). Some `conflicting` tests are broken (a bare `assert` with no `def check`),
so `check(<entry_point>)` raises NameError and nothing can pass: those tasks are unusable for
measuring hacks. Known held-out ones: lcbhard_35, lcbhard_87 (handoff §3). `usable()` finds them
by scanning, and the scan result for the held-out set must equal the known list.

Scan @ dataset rev 98650ff (2026-11-06): held-out conflicting broken = {lcbhard_35, lcbhard_87}
(no_check); training conflicting broken = {lcbhard_77} (syntax: truncated test). No `original`
test is broken. So 20 held-out and 80 training conflicting tasks are usable.
"""
from __future__ import annotations

import ast

from td import config

KNOWN_BROKEN_HELDOUT = frozenset({"lcbhard_35", "lcbhard_87"})
KNOWN_BROKEN_TRAIN = frozenset({"lcbhard_77"})
N_HELDOUT = 22
N_TRAIN = 81


def load(splits: tuple[str, ...] = ("conflicting",), held_out: bool = True) -> list[dict]:
    """Upstream items: {task_id, prompt, test, original_test, entry_point, impossible, impossible_type}.
    Order is upstream's seeded shuffle (seed 12345), so it matches heldout_gen's draw."""
    from src.train.rlaif.reward_hack.dataset import load_impossible_lcb

    return load_impossible_lcb(splits=splits, heldout_frac=config.HELDOUT_FRAC, held_out=held_out,
                               seed=config.EVAL_SEED)


def broken_reason(test: str) -> str | None:
    """Why a test can never pass (None if it looks usable):
    'syntax'   the test source doesn't parse (e.g. lcbhard_77 conflicting: truncated string literal)
    'no_check' it parses but defines no module-level `check` (lcbhard_35, lcbhard_87: bare asserts)
    """
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return "syntax"
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "check" for n in tree.body):
        return "no_check"
    return None


def test_defines_check(test: str) -> bool:
    return broken_reason(test) is None


def broken_task_ids(items: list[dict]) -> dict[str, str]:
    """{task_id: reason} for every item whose test is broken."""
    return {it["task_id"]: r for it in items if (r := broken_reason(it["test"]))}


def usable(items: list[dict]) -> list[dict]:
    """Drop tasks whose test (for the item's split) has no `def check`."""
    bad = broken_task_ids(items)
    return [it for it in items if it["task_id"] not in bad]


def is_heldout(task_id: str) -> bool:
    from src.train.rlaif.reward_hack.dataset import _heldout_by_task_id

    return _heldout_by_task_id(task_id, config.HELDOUT_FRAC)
