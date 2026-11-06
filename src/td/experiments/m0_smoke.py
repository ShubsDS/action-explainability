"""M0 smoke test (cap $10): one rollout from anti SFT and one from anti_s1 step 60, graded in Docker.

    bin/td python -m td.experiments.m0_smoke --plan    # spend nothing
    bin/td python -m td.experiments.m0_smoke           # ~2 rollouts; worst case ≈ $0.20

If the public Tinker sampler paths don't load from our account, STOP and tell the user
(serving the 120B model from the HF adapters isn't budgeted).
"""
from __future__ import annotations

import argparse
import asyncio
import json

from td import checkpoints, config, cost, tasks
from td.experiments.common import describe, print_plan

STORE = "m0_smoke"


def _prompt_bound(item: dict) -> int:
    # Upper bound on prompt tokens without loading the tokenizer: every token is >= 1 byte.
    return len((config.SYS_PROMPT + item["prompt"] + item["test"]).encode()) + 256


async def main(plan: bool) -> None:
    ckpts = [checkpoints.sft("anti"), checkpoints.get("anti", 1, 60)]
    item = tasks.usable(tasks.load(("conflicting",), held_out=True))[0]
    budget = cost.Budget("M0")
    print(f"task {item['task_id']} (held-out, conflicting)")
    print_plan(budget, ckpts, len(ckpts), _prompt_bound(item))
    if plan:
        return
    from td.sampling import Sampler, SampleSpec

    rows = []
    for ck in ckpts:
        try:
            s = Sampler(ck, budget, STORE, concurrency=1)
            rows.append(await s.run(SampleSpec(item=item, sample_idx=0)))
        except Exception as e:  # surface checkpoint-access failures plainly (handoff M0 stop rule)
            print(f"FAILED on {ck.label}: {type(e).__name__}: {e}")
            print("If this is a checkpoint-access error, stop and tell the user.")
            raise
    for r in rows:
        g = r["grade"]
        print(json.dumps({"checkpoint": r["checkpoint"], "task": r["task_id"], "truncated": r["truncated"],
                          "n_prompt_tokens": r["n_prompt_tokens"], "n_output_tokens": r["n_output_tokens"],
                          "hacked": g.get("hacked_on_impossible"), "passes_original": g.get("passes_original"),
                          "cls": r["cls"], "cost_usd": round(r["cost_usd"], 5)}))
    print("summary:", describe(rows))
    print("budget:", {k: v for k, v in budget.status().items() if k != "open_rids"})
    print("Now: record measured tokens/rollout and re-checked prices in NOTES.md.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true")
    asyncio.run(main(ap.parse_args().plan))
