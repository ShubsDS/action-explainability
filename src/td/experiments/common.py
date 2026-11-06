"""Shared helpers for milestone scripts: cell sampling with a cost plan."""
from __future__ import annotations

import statistics

from td import config, cost
from td.checkpoints import Checkpoint


def plan_cost(n_rollouts: int, n_prompt_tokens: int, max_tokens: int = config.MAX_TOKENS) -> float:
    """Worst case for n first attempts (every one at the full budget)."""
    return n_rollouts * cost.tinker_rollout_worst_case(n_prompt_tokens, max_tokens)


def print_plan(budget: cost.Budget, ckpts: list[Checkpoint], n_rollouts: int, n_prompt_tokens: int) -> float:
    worst = plan_cost(n_rollouts, n_prompt_tokens)
    s = budget.status()
    print(f"milestone {budget.milestone}: cap ${s['cap']:.2f}, settled ${s['settled']:.4f}, "
          f"outstanding ${s['outstanding']:.4f}, remaining ${s['remaining']:.4f}")
    for c in ckpts:
        print(f"  checkpoint {c.label}: {c.sampler_path}")
    print(f"  {n_rollouts} rollouts; worst case ${worst:.2f} "
          f"(prompt ≈{n_prompt_tokens} tok, {config.MAX_TOKENS} output tok each, uncached)")
    if worst > s["remaining"]:
        print("  WORST CASE EXCEEDS REMAINING CAP: run would stop partway (resumable); ask the user before raising it")
    return worst


def describe(rows: list[dict]) -> dict:
    n = len(rows)
    out_tok = [r["n_output_tokens"] for r in rows]
    return {
        "n": n,
        "hacked": sum(bool(r["grade"].get("hacked_on_impossible")) for r in rows),
        "truncated": sum(bool(r["truncated"]) for r in rows),
        "accidental_pass": sum(bool(r["accidental_pass"]) for r in rows),
        "mean_output_tokens": statistics.fmean(out_tok) if out_tok else float("nan"),
        "median_output_tokens": statistics.median(out_tok) if out_tok else float("nan"),
        "mean_prompt_tokens": statistics.fmean(r["n_prompt_tokens"] for r in rows) if rows else float("nan"),
        "cost_usd": sum(r["cost_usd"] for r in rows),
    }
