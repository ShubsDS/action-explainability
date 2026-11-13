"""M5 step 1 (redesign of 2026-11-13): does the reasoning have any causal role in the hack?

At anti_s1@30 the reasoning almost never mentions the hack, and the hack is one line in a short
answer. So instead of forking early and paying for a full continuation, every arm here fixes the
reasoning, closes it, and samples ONLY the answer (a few hundred tokens, never truncated by the
context limit). The arms, their sizes and the outcome are in prereg/m5.yaml (`reasoning_role`);
nothing is hard-coded here.

  hacking checkpoint (rl):   none   the reasoning block is closed at once
                             cutNN  the parent's reasoning up to NN%, cut at a paragraph break
                             own    the parent's whole reasoning (the answer is resampled)
                             donor  an honest step-0 rollout's whole reasoning, same task
  honest checkpoint (sft):   none, own, and rl_reasoning (the hacking parent's whole reasoning)

Outcome: td.grading.attempted_hack on completed answers. Budget: the M5a cap.

    bin/td python -m td.experiments.m5_reasoning --plan
    bin/td python -m td.experiments.m5_reasoning --stage donors --pilot
    bin/td python -m td.experiments.m5_reasoning --stage arms --pilot
    bin/td python -m td.experiments.m5_reasoning --stage analyze --pilot
Drop --pilot for the main run; that needs `status: frozen` in prereg/m5.yaml.
"""
from __future__ import annotations

import argparse
import asyncio
import collections

import numpy as np
import yaml

from td import checkpoints, config, cost, grading, render, store, tasks
from td.analysis.stats import N_BOOT, cluster_bootstrap, wilson

MILESTONE = "M5a"


class NotFrozen(RuntimeError):
    pass


def load_cfg(path=None) -> dict:
    doc = yaml.safe_load(open(path or config.PREREG_DIR / "m5.yaml"))
    return {"status": doc["status"], **doc["reasoning_role"]}


def require_frozen(cfg: dict, pilot: bool) -> None:
    if not pilot and cfg["status"] != "frozen":
        raise NotFrozen("main sampling needs `status: frozen` in prereg/m5.yaml (and its git tag); "
                        "use --pilot until then")


def _ckpt(cell: dict):
    return checkpoints.get(cell["character"], cell.get("seed"), cell["step"])


def _cond(arm: str, pilot: bool) -> str:
    return f"pilot:{arm}" if pilot else arm


# ---------------------------------------------------------------- what to sample

def usable_items() -> dict[str, dict]:
    """Held-out conflicting tasks whose test can be passed at all, by task id."""
    return {it["task_id"]: it for it in tasks.usable(tasks.load(("conflicting",), held_out=True))}


def completed(rows, items: dict, condition: str) -> list[dict]:
    """First attempts on usable tasks that finished with a closed reasoning block, in a fixed order."""
    out = [r for r in rows if r["condition"] == condition and r["task_id"] in items
           and not r["truncated"] and render.think_span(r["output_tokens"]) is not None]
    return sorted(out, key=lambda r: (r["task_id"], r["sample_idx"]))


def load_parents(cfg: dict, items: dict) -> list[dict]:
    return completed(store.RolloutStore(cfg["cells"]["rl"]["parent_store"]).rows(), items, "base")


def load_donors(cfg: dict, items: dict, pilot: bool) -> list[dict]:
    """Honest donors: completed step-0 first attempts that did not try to hack."""
    rows = store.RolloutStore(cfg["cells"]["honest"]["donor_store"]).rows()
    return [r for r in completed(rows, items, "pilot" if pilot else "base") if not grading.attempted_hack(r)]


def _one_per_task(rows: list[dict], n: int) -> list[dict]:
    """The first row of each of the first n tasks (rows are already in task order)."""
    seen: dict[str, dict] = {}
    for r in rows:
        seen.setdefault(r["task_id"], r)
    return list(seen.values())[:n]


def _through_close(row: dict, close: list[int]) -> int:
    """Output tokens up to and including the row's own `</think>`."""
    return render.think_span(row["output_tokens"])[1] + len(close)


def build_specs(cfg: dict, items: dict, parents: list[dict], donors: list[dict], pilot: bool) -> dict:
    """{"rl": [SampleSpec], "sft": [SampleSpec], "skipped": {...}} for every arm.

    A parent or donor whose whole reasoning leaves less than `min_answer_room` tokens of context
    is dropped from EVERY arm, so all arms of one checkpoint share the same parents."""
    from td.sampling import SampleSpec

    close = render.close_think_tokens()
    n_per = cfg["pilot"]["samples"] if pilot else cfg["samples_per_parent"]
    n_none = cfg["pilot"]["samples"] if pilot else cfg["samples_none_per_task"]
    room, cap = cfg["min_answer_room"], cfg["answer_max_tokens"]

    def fits(row):
        return config.MAX_TOKENS - (len(row["prompt_tokens"]) + _through_close(row, close)) >= room

    skipped = {"parents_too_long": sum(not fits(p) for p in parents),
               "donors_too_long": sum(not fits(d) for d in donors)}
    parents, donors = [p for p in parents if fits(p)], [d for d in donors if fits(d)]
    if pilot:
        parents, donors = _one_per_task(parents, cfg["pilot"]["parents"]), _one_per_task(donors, cfg["pilot"]["parents"])
    task_ids = sorted({p["task_id"] for p in parents})

    def spec(arm, item, parent, k, insert, n):
        src = parent or {"prompt_tokens": prompts[item["task_id"]], "output_tokens": []}
        prefix, budget = render.fork_prefix(src, k, insert)
        return [SampleSpec(item=item, sample_idx=j, condition=_cond(arm, pilot),
                           parent_rollout_id=parent["rollout_id"] if parent else None, fork_token=k,
                           insert_id="close" if insert else None, prompt_tokens=prefix,
                           output_prefix=prefix[len(src["prompt_tokens"]):], max_tokens=min(cap, budget))
                for j in range(n)]

    prompts = {p["task_id"]: p["prompt_tokens"] for p in parents}  # same prompt at every checkpoint
    rl, sft = [], []
    for t in task_ids:
        rl += spec("none", items[t], None, 0, close, n_none)
        sft += spec("none", items[t], None, 0, close, n_none)
    for p in parents:
        item, n_reason = items[p["task_id"]], render.think_span(p["output_tokens"])[1]
        paras = render.paragraph_boundaries(p["output_tokens"], n_reason)
        lines = render.paragraph_boundaries(p["output_tokens"], n_reason, ending="\n")
        for arm, frac in cfg["cuts"].items():
            k = render.paragraph_cut(n_reason, frac, paras, lines, cfg["cut_tolerance"])
            rl += spec(arm, item, p, k, close, n_per)
        whole = _through_close(p, close)
        rl += spec("own", item, p, whole, None, n_per)
        sft += spec("rl_reasoning", item, p, whole, None, n_per)
    for d in donors:
        if d["task_id"] not in prompts:
            continue
        item, whole = items[d["task_id"]], _through_close(d, close)
        rl += spec("donor", item, d, whole, None, n_per)
        sft += spec("own", item, d, whole, None, n_per)
    return {"rl": rl, "sft": sft, "skipped": skipped, "n_parents": len(parents), "n_donors": len(donors)}


def donor_specs(cfg: dict, items: dict, pilot: bool) -> list:
    """Step-0 first attempts under upstream settings: the pool the honest donors come from."""
    from td.sampling import SampleSpec

    ids = sorted(items)[:cfg["pilot"]["donor_tasks"]] if pilot else sorted(items)
    n = cfg["pilot"]["samples"] if pilot else cfg["donor_samples_per_task"]
    return [SampleSpec(item=items[t], sample_idx=j, condition="pilot" if pilot else "base")
            for t in ids for j in range(n)]


# ---------------------------------------------------------------- cost

def spec_costs(specs: list, answer_tokens: int) -> tuple[float, float]:
    """(worst case, expected) USD: prefill uncached, output at max_tokens vs a typical answer."""
    worst = sum(cost.tinker_rollout_worst_case(len(s.prompt_tokens), s.max_tokens) for s in specs)
    expected = sum(cost.tinker_cost(len(s.prompt_tokens), min(answer_tokens, s.max_tokens)) for s in specs)
    return worst, expected


def print_plan(cfg: dict, pilot: bool) -> None:
    items = usable_items()
    budget = cost.Budget(MILESTONE)
    s = budget.status()
    print(f"prereg/m5.yaml status: {cfg['status']}; {'PILOT' if pilot else 'main'} plan")
    print(f"milestone {MILESTONE}: cap ${s['cap']:.2f}, settled ${s['settled']:.2f}, remaining ${s['remaining']:.2f}"
          + ("" if config.KILL_RULE_SETTLED else "  [blocked by the kill rule]"))
    parents, donors = load_parents(cfg, items), load_donors(cfg, items, pilot)
    dn = donor_specs(cfg, items, pilot)
    n_prompt = {p["task_id"]: len(p["prompt_tokens"]) for p in parents}  # prompts are the same at step 0
    dn_worst = sum(cost.tinker_rollout_worst_case(n_prompt.get(sp.item["task_id"], 4096), sp.max_tokens) for sp in dn)
    print(f"donor first attempts ({_ckpt(cfg['cells']['honest']).label}): {len(dn)} rollouts, worst case "
          f"${dn_worst:.2f}; on disk: {len(donors)} honest donors")
    b = build_specs(cfg, items, parents, donors, pilot)
    print(f"parents {b['n_parents']} (of {len(parents)} completed), donors {b['n_donors']}; skipped {b['skipped']}")
    total_w = total_e = 0.0
    for side in ("rl", "sft"):
        by = collections.defaultdict(list)
        for sp in b[side]:
            by[sp.condition].append(sp)
        for arm, specs in by.items():
            w, e = spec_costs(specs, cfg["typical_answer_tokens"])
            total_w, total_e = total_w + w, total_e + e
            print(f"  {_ckpt(cfg['cells']['rl' if side == 'rl' else 'honest']).label:12} {arm:20} "
                  f"{len(specs):5d} samples  expected ${e:6.2f}  worst ${w:6.2f}")
    print(f"arms total: expected ${total_e:.2f}, worst case ${total_w:.2f} (prefill billed uncached)")
    if not donors:
        print("no donors on disk yet: the donor / sft-own arms are empty until `--stage donors` has run")


# ---------------------------------------------------------------- analysis

def arm_table(rows: list[dict]) -> dict:
    """Per arm: n, truncated, completed, attempts, hacks, and the attempt rate among completed
    answers with the wider of a task-cluster bootstrap and a Wilson interval (as upstream does)."""
    out = {}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    for arm, rs in sorted(by.items()):
        done = [r for r in rs if not r["truncated"]]
        vals = [float(grading.attempted_hack(r)) for r in done]
        k = int(sum(vals))
        row = {"n": len(rs), "truncated": len(rs) - len(done), "completed": len(done), "attempted": k,
               "hacked": sum(bool(r["grade"].get("hacked_on_impossible")) for r in done),
               "n_tasks": len({r["task_id"] for r in done})}
        if done:
            bs = cluster_bootstrap(vals, [r["task_id"] for r in done])
            wl, wh = wilson(k, len(done))
            row.update(rate=k / len(done), lo=min(bs["lo"], wl), hi=max(bs["hi"], wh))
        out[arm] = row
    return out


def contrast(rows: list[dict], arm: str, base: str, n_boot: int = N_BOOT, seed: int = 0, conf: float = 0.90) -> dict:
    """rate(arm) − rate(base) among completed answers, resampling TASKS (both arms together).
    A 90% two-sided interval, so each end is a one-sided 95% bound."""
    per = collections.defaultdict(lambda: {arm: [0, 0], base: [0, 0]})
    for r in rows:
        if r["condition"] in (arm, base) and not r["truncated"]:
            c = per[r["task_id"]][r["condition"]]
            c[0] += grading.attempted_hack(r)
            c[1] += 1
    ts = sorted(per)
    a = np.array([[per[t][x][0], per[t][x][1]] for t in ts for x in (arm, base)], dtype=float).reshape(len(ts), 2, 2)

    def diff(m):
        tot = m.sum(axis=0)
        return tot[0, 0] / tot[0, 1] - tot[1, 0] / tot[1, 1] if tot[0, 1] and tot[1, 1] else np.nan

    rng = np.random.default_rng(seed)
    draws = np.array([diff(a[rng.integers(0, len(ts), len(ts))]) for _ in range(n_boot)])
    lo, hi = np.nanquantile(draws, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return {"diff": float(diff(a)), "lo": float(lo), "hi": float(hi), "n_tasks": len(ts)}


def analyze(cfg: dict, pilot: bool) -> None:
    pre = "pilot:" if pilot else ""
    for side, cell in (("rl", cfg["cells"]["rl"]), ("sft", cfg["cells"]["honest"])):
        rows = [dict(r, condition=r["condition"][len(pre):]) for r in store.RolloutStore(cell["arm_store"]).rows()
                if r["condition"].startswith("pilot:") == pilot]
        print(f"\n{_ckpt(cell).label}: attempted-hack rate among completed answers")
        for arm, t in arm_table(rows).items():
            rate = f"{100 * t['rate']:5.1f}% [{100 * t['lo']:.1f}, {100 * t['hi']:.1f}]" if t["completed"] else "n/a"
            print(f"  {arm:14} {rate:26} {t['attempted']}/{t['completed']} completed, {t['truncated']} truncated, "
                  f"{t['hacked']} hacked, {t['n_tasks']} tasks")
        for arm in cfg["contrasts"][side]:
            if any(r["condition"] == arm for r in rows) and any(r["condition"] == "own" for r in rows):
                c = contrast(rows, arm, "own")
                print(f"  {arm} − own: {100 * c['diff']:+.1f} pp, 90% interval [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}] "
                      f"({c['n_tasks']} tasks)")


# ---------------------------------------------------------------- run

async def run(stage: str, pilot: bool) -> None:
    cfg = load_cfg()
    if stage == "plan":
        return print_plan(cfg, pilot)
    if stage == "analyze":
        return analyze(cfg, pilot)
    require_frozen(cfg, pilot)
    from td.sampling import Sampler

    items, budget = usable_items(), cost.Budget(MILESTONE)
    jobs = []
    if stage == "donors":
        jobs.append((cfg["cells"]["honest"], cfg["cells"]["honest"]["donor_store"], donor_specs(cfg, items, pilot)))
    else:
        b = build_specs(cfg, items, load_parents(cfg, items), load_donors(cfg, items, pilot), pilot)
        print(f"parents {b['n_parents']}, donors {b['n_donors']}, skipped {b['skipped']}")
        jobs += [(cfg["cells"]["rl"], cfg["cells"]["rl"]["arm_store"], b["rl"]),
                 (cfg["cells"]["honest"], cfg["cells"]["honest"]["arm_store"], b["sft"])]
    for cell, store_name, specs in jobs:
        res = await Sampler(_ckpt(cell), budget, store_name).run_all(specs)
        errs = [r for r in res if isinstance(r, BaseException)]
        for e in errs[:5]:
            print("ERROR:", type(e).__name__, e)
        print(f"{_ckpt(cell).label} → {store_name}: {len(res) - len(errs)}/{len(specs)} rows"
              + (f", {len(errs)} errors; rerun to resume (stored rows are free)" if errs else ""))
    print("budget:", {k: v for k, v in budget.status().items() if k != "open_rids"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the samples and cost, spend nothing")
    ap.add_argument("--stage", choices=("donors", "arms", "analyze"))
    ap.add_argument("--pilot", action="store_true", help="small run, stored under pilot conditions")
    a = ap.parse_args()
    if not a.plan and not a.stage:
        ap.error("pass --plan or --stage")
    asyncio.run(run("plan" if a.plan else a.stage, a.pilot))
