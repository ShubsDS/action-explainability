"""M5 step 2: does what is said in the CONVERSATION change whether the model hacks?

Step 1 (td.experiments.m5_reasoning) varies a statement inside the reasoning. This step varies
what is said outside it, where the Corin character actually lives: a sentence from the user, or
the model's own earlier turn. Every arm is a full first attempt at the hacking checkpoint; arms,
texts and sizes are in prereg/m5.yaml (`conversation_ladder`).

  baseline          M1's first attempts, reused (never re-sampled)
  placebo_turn      an unrelated question and answer, then the task
  own_promise       the model is asked about `check`, has answered that it will not touch it, then the task
  user_placebo      the task, plus one unrelated sentence
  user_prohibition  the task, plus one sentence forbidding any change to `check`

Two more cells give the scale: the same checkpoint on SOLVABLE tasks (is the override conditional
on the test being impossible at all?), and a floor checkpoint that went through RL without
learning to hack.

    bin/td python -m td.experiments.m5_ladder --plan [--pilot]
    bin/td python -m td.experiments.m5_ladder --stage arms|solvable|floor --pilot
    bin/td python -m td.experiments.m5_ladder --stage analyze [--pilot]
Drop --pilot for the main run; that needs `status: frozen` in prereg/m5.yaml.
"""
from __future__ import annotations

import argparse
import asyncio
import collections

import yaml

from td import config, cost, render, store, tasks
from td.experiments.m5_reasoning import (MILESTONE, _ckpt, _cond, arm_table, contrast, require_frozen,
                                         usable_items, verdict, working_hack)


def load_cfg(path=None) -> dict:
    doc = yaml.safe_load(open(path or config.PREREG_DIR / "m5.yaml"))
    return {"status": doc["status"], **doc["conversation_ladder"]}


def _subset(ids: list[str], cfg: dict, pilot: bool) -> list[str]:
    return ids[:cfg["pilot"]["tasks"]] if pilot else ids


def _n(cfg: dict, pilot: bool) -> int:
    return cfg["pilot"]["samples"] if pilot else cfg["samples_per_task"]


def arm_specs(cfg: dict, items: dict, pilot: bool) -> list:
    """Full first attempts for every conversation arm, on the usable held-out tasks."""
    from td.sampling import SampleSpec

    specs = []
    for t in _subset(sorted(items), cfg, pilot):
        for arm, a in cfg["arms"].items():
            before = [{"role": m["role"], "content": m["content"].strip()} for m in a.get("before", [])]
            suffix = ("\n\n" + a["user_suffix"].strip()) if a.get("user_suffix") else ""
            prompt, message = render.conversation_prompt(items[t], before=before, user_suffix=suffix)
            specs += [SampleSpec(item=items[t], sample_idx=j, condition=_cond(arm, pilot),
                                 prompt_tokens=prompt, user_prompt=message) for j in range(_n(cfg, pilot))]
    return specs


def plain_specs(cfg: dict, items: dict, pilot: bool) -> list:
    """Ordinary first attempts (upstream's prompt, rendered by the sampler) on `items`."""
    from td.sampling import SampleSpec

    return [SampleSpec(item=items[t], sample_idx=j, condition="pilot" if pilot else "base")
            for t in _subset(sorted(items), cfg, pilot) for j in range(_n(cfg, pilot))]


def solvable_items() -> dict[str, dict]:
    """The held-out tasks with their ORIGINAL (passable) tests."""
    return {it["task_id"]: it for it in tasks.load(("original",), held_out=True)}


def jobs(cfg: dict, stage: str, pilot: bool) -> list[tuple[dict, str, list]]:
    """(checkpoint cell, store name, specs) for one stage."""
    c = cfg["cells"]
    if stage == "arms":
        return [(c["rl"], c["rl"]["arm_store"], arm_specs(cfg, usable_items(), pilot))]
    if stage == "solvable":
        return [(c["rl"], c["rl"]["solvable_store"], plain_specs(cfg, solvable_items(), pilot))]
    return [(c["floor"], c["floor"]["store"], plain_specs(cfg, usable_items(), pilot))]


def print_plan(cfg: dict, pilot: bool) -> None:
    budget = cost.Budget(MILESTONE)
    s = budget.status()
    print(f"prereg/m5.yaml status: {cfg['status']}; {'PILOT' if pilot else 'main'} plan")
    print(f"milestone {MILESTONE}: cap ${s['cap']:.2f}, settled ${s['settled']:.2f}, remaining ${s['remaining']:.2f}"
          + ("" if config.KILL_RULE_SETTLED else "  [blocked by the kill rule]"))
    total_w = total_e = 0.0
    for stage in ("arms", "solvable", "floor"):
        for cell, store_name, specs in jobs(cfg, stage, pilot):
            by = collections.Counter(sp.condition for sp in specs)
            # a plain first attempt is rendered by the sampler; bound its prompt as M1 did (4,096 covers every task)
            worst = sum(cost.tinker_rollout_worst_case(len(sp.prompt_tokens) if sp.prompt_tokens else 4096) for sp in specs)
            exp = len(specs) * cfg["typical_rollout_usd"]
            total_w, total_e = total_w + worst, total_e + exp
            print(f"  {stage:9} {_ckpt(cell).label:12} → {store_name}: {len(specs)} rollouts {dict(by)}  "
                  f"expected ${exp:.2f}  worst ${worst:.2f}")
    print(f"total: expected ${total_e:.2f} (at ${cfg['typical_rollout_usd']} per rollout, measured at anti_s1@30), "
          f"worst case ${total_w:.2f}")


def _rows(store_name: str, pilot: bool, rename: dict | None = None) -> list[dict]:
    """Rows of a store for this run (pilot or main), with the `pilot` marker removed and plain
    first attempts renamed (e.g. base -> baseline)."""
    out = []
    for r in store.RolloutStore(store_name).rows():
        cond = r["condition"]
        is_pilot = cond == "pilot" or cond.startswith("pilot:")
        if is_pilot != pilot:
            continue
        cond = "base" if cond == "pilot" else cond.removeprefix("pilot:")
        out.append(dict(r, condition=(rename or {}).get(cond, cond)))
    return out


def analyze(cfg: dict, pilot: bool) -> None:
    c, usable = cfg["cells"], usable_items()
    baseline = [dict(r, condition="baseline") for r in store.RolloutStore(c["rl"]["baseline_store"]).rows()
                if r["condition"] == "base" and r["task_id"] in usable]
    rows = baseline + _rows(c["rl"]["arm_store"], pilot)
    print(f"{_ckpt(c['rl']).label}: attempted-hack rate among answered samples"
          + (" (baseline = M1 main data, shown for scale)" if pilot else ""))
    _print_arms(rows)
    present = {r["condition"] for r in rows}
    for arm, base in cfg["contrasts"].items():
        if arm in present and base in present:
            k = contrast(rows, arm, base)
            primary = [arm, base] in cfg["primary"]
            print(f"  {arm} − {base}: {100 * k['diff']:+.1f} pp, 90% interval [{100 * k['lo']:+.1f}, {100 * k['hi']:+.1f}], "
                  f"{k['n_tasks']} tasks ({k['n_tasks_dropped']} dropped): {verdict(k, cfg['decision']['margin_pp'])}"
                  + (" [primary]" if primary else ""))
            if primary:
                w = contrast(rows, arm, base, outcome=working_hack)
                print(f"    working hacks only: {100 * w['diff']:+.1f} pp [{100 * w['lo']:+.1f}, {100 * w['hi']:+.1f}]: "
                      f"{verdict(w, cfg['decision']['margin_pp'])}")
    for label, cell, name in (("solvable tasks", c["rl"], c["rl"]["solvable_store"]), ("floor", c["floor"], c["floor"]["store"])):
        print(f"\n{_ckpt(cell).label}, {label}:")
        _print_arms(_rows(name, pilot, {"base": label}))


def _print_arms(rows: list[dict]) -> None:
    for arm, t in arm_table(rows).items():
        rate = f"{100 * t['rate']:5.1f}% [{100 * t['lo']:.1f}, {100 * t['hi']:.1f}]" if t["answered"] else "n/a"
        print(f"  {arm:17} {rate:26} {t['attempted']}/{t['answered']} answered; {t['truncated']} truncated, "
              f"{t['no_code']} no code; {t['hacked']} hacked; {t['n_tasks']} tasks")
    if not rows:
        print("  (no rows yet)")


async def run(stage: str, pilot: bool) -> None:
    cfg = load_cfg()
    if stage == "plan":
        return print_plan(cfg, pilot)
    if stage == "analyze":
        return analyze(cfg, pilot)
    require_frozen(cfg, pilot)
    from td.sampling import Sampler

    budget = cost.Budget(MILESTONE)
    for cell, store_name, specs in jobs(cfg, stage, pilot):
        res = await Sampler(_ckpt(cell), budget, store_name).run_all(specs)
        errs = [r for r in res if isinstance(r, BaseException)]
        for e in errs[:5]:
            print("ERROR:", type(e).__name__, e)
        print(f"{_ckpt(cell).label} → {store_name}: {len(res) - len(errs)}/{len(specs)} rows"
              + (f", {len(errs)} errors; rerun to resume (stored rows are free)" if errs else ""))
    print("budget:", {k: v for k, v in budget.status().items() if k != "open_rids"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the rollouts and cost, spend nothing")
    ap.add_argument("--stage", choices=("arms", "solvable", "floor", "analyze"))
    ap.add_argument("--pilot", action="store_true", help="small run, stored under pilot conditions")
    a = ap.parse_args()
    if not a.plan and not a.stage:
        ap.error("pass --plan or --stage")
    asyncio.run(run("plan" if a.plan else a.stage, a.pilot))
