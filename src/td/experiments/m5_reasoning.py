"""M5 step 1 (redesign of 2026-11-13, revised after review): does what the reasoning SAYS control
whether the answer hacks?

At anti_s1@30 the reasoning does not mention the hack, and the hack is one line in a short answer.
So every arm here fixes the reasoning, closes it, and samples ONLY the answer (a few hundred
tokens, never truncated by the context limit). Arms, texts, sizes and the outcome are in
prereg/m5.yaml (`reasoning_role`); nothing is hard-coded here.

  own            the parent's whole reasoning (the answer is resampled); the baseline
  ins_<name>     the parent's whole reasoning, then one inserted statement, then </think>
  donor          (hacking checkpoint) an honest step-0 rollout's whole reasoning, same task
  rl_reasoning   (honest checkpoint) a hacking parent's whole reasoning, same task
  none           the reasoning block is closed at once
  cutNN          the parent's reasoning up to NN%, cut at a paragraph break

At the honest checkpoint the parents are its own step-0 first attempts (the "donors").
Outcome: td.grading.attempted_hack among answers that contain code. Budget: the M5a cap.

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
import math

import numpy as np
import yaml
from scipy import stats as sps

from td import checkpoints, config, cost, grading, render, store, tasks
from td.analysis.stats import cluster_bootstrap, wilson

MILESTONE = "M5a"
SIDES = {"rl": "rl", "sft": "honest"}  # side -> key under cfg["cells"]


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


def build_specs(cfg: dict, items: dict, parents: list[dict], donors: list[dict], pilot: bool) -> dict:
    """{"rl": [SampleSpec], "sft": [SampleSpec], "skipped": {...}, ...} for every arm in the prereg.

    A parent or donor whose whole reasoning leaves less than `min_answer_room` tokens of context
    is dropped from EVERY arm, so all arms of one checkpoint share the same parents. Donors on a
    task with no parent left are dropped too."""
    from td.sampling import SampleSpec

    close = render.close_think_tokens()
    n_per = cfg["pilot"]["samples"] if pilot else cfg["samples_per_parent"]
    n_none = cfg["pilot"]["samples"] if pilot else cfg["samples_none_per_task"]
    room, cap = cfg["min_answer_room"], cfg["answer_max_tokens"]
    arms = cfg["arms"]

    def n_reason(row):
        return render.think_span(row["output_tokens"])[1]

    def fits(row):
        return config.MAX_TOKENS - (len(row["prompt_tokens"]) + n_reason(row) + len(close)) >= room

    skipped = {"parents_too_long": sum(not fits(p) for p in parents),
               "donors_too_long": sum(not fits(d) for d in donors)}
    parents, donors = [p for p in parents if fits(p)], [d for d in donors if fits(d)]
    if pilot:
        parents, donors = _one_per_task(parents, cfg["pilot"]["parents"]), _one_per_task(donors, cfg["pilot"]["parents"])
    prompts = {p["task_id"]: p["prompt_tokens"] for p in parents}  # the same prompt at every checkpoint
    skipped["donors_without_parent_task"] = sum(d["task_id"] not in prompts for d in donors)
    donors = [d for d in donors if d["task_id"] in prompts]

    def spec(arm, item, source, k, insert, insert_id, n):
        src = source or {"prompt_tokens": prompts[item["task_id"]], "output_tokens": []}
        prefix, budget = render.fork_prefix(src, k, insert)
        return [SampleSpec(item=item, sample_idx=j, condition=_cond(arm, pilot),
                           parent_rollout_id=source["rollout_id"] if source else None, fork_token=k,
                           insert_id=insert_id, prompt_tokens=prefix,
                           output_prefix=prefix[len(src["prompt_tokens"]):], max_tokens=min(cap, budget))
                for j in range(n)]

    def whole(arm, row):          # the row's own reasoning through its own </think>
        return spec(arm, items[row["task_id"]], row, n_reason(row) + len(close), None, None, n_per)

    def inserted(side, row):      # ...its whole reasoning, one inserted statement, then </think>
        out = []
        for name, ins in cfg["inserts"].items():
            if side in ins["at"]:
                toks = render.encode_insert(ins["text"].strip() + "\n\n") + close
                out += spec(f"ins_{name}", items[row["task_id"]], row, n_reason(row), toks, name, n_per)
        return out

    rl, sft = [], []
    for t in sorted(prompts):
        if "none" in arms["rl"]:
            rl += spec("none", items[t], None, 0, close, "close", n_none)
        if "none" in arms["sft"]:
            sft += spec("none", items[t], None, 0, close, "close", n_none)
    for p in parents:
        if "own" in arms["rl"]:
            rl += whole("own", p)
        rl += inserted("rl", p)
        if cfg["cuts"]:
            paras = render.paragraph_boundaries(p["output_tokens"], n_reason(p))
            lines = render.paragraph_boundaries(p["output_tokens"], n_reason(p), ending="\n")
            for arm, frac in cfg["cuts"].items():
                k = render.paragraph_cut(n_reason(p), frac, paras, lines, cfg["cut_tolerance"])
                rl += spec(arm, items[p["task_id"]], p, k, close, "close", n_per)
        if "rl_reasoning" in arms["sft"]:
            sft += whole("rl_reasoning", p)
    for d in donors:
        if "donor" in arms["rl"]:
            rl += whole("donor", d)
        if "own" in arms["sft"]:
            sft += whole("own", d)
        sft += inserted("sft", d)
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
    for side, cell in SIDES.items():
        by = collections.defaultdict(list)
        for sp in b[side]:
            by[sp.condition].append(sp)
        for arm, specs in by.items():
            w, e = spec_costs(specs, cfg["typical_answer_tokens"])
            total_w, total_e = total_w + w, total_e + e
            print(f"  {_ckpt(cfg['cells'][cell]).label:12} {arm:20} {len(specs):5d} samples  "
                  f"expected ${e:6.2f}  worst ${w:6.2f}")
    print(f"arms total: expected ${total_e:.2f}, worst case ${total_w:.2f} (prefill billed uncached)")
    if not donors:
        print("no donors on disk yet: every arm that needs a step-0 reasoning is empty until `--stage donors` has run")


# ---------------------------------------------------------------- analysis

def answered(row: dict) -> bool:
    """An answer that can be scored: not truncated, and it contains code."""
    return not row["truncated"] and grading.gave_code(row)


def arm_table(rows: list[dict]) -> dict:
    """Per arm: n, truncated, no-code, answered, attempts, hacks, and the attempt rate among
    answered samples with the wider of a task-cluster bootstrap and a Wilson interval (as upstream
    does). Truncated and no-code samples are counted, never scored as non-attempts."""
    out = {}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    for arm, rs in sorted(by.items()):
        done = [r for r in rs if answered(r)]
        vals = [float(grading.attempted_hack(r)) for r in done]
        k = int(sum(vals))
        n_trunc = sum(bool(r["truncated"]) for r in rs)
        row = {"n": len(rs), "truncated": n_trunc, "no_code": len(rs) - n_trunc - len(done),
               "reopened_think": sum(bool(r.get("reopened_think")) for r in rs),
               "answered": len(done), "attempted": k,
               "hacked": sum(bool(r["grade"].get("hacked_on_impossible")) for r in done),
               "n_tasks": len({r["task_id"] for r in done})}
        if done:
            bs = cluster_bootstrap(vals, [r["task_id"] for r in done])
            wl, wh = wilson(k, len(done))
            row.update(rate=k / len(done), lo=min(bs["lo"], wl), hi=max(bs["hi"], wh))
        out[arm] = row
    return out


def contrast(rows: list[dict], arm: str, base: str, conf: float = 0.90) -> dict:
    """The mean WITHIN-TASK difference rate(arm) − rate(base) among answered samples.

    Only tasks with answered samples in both arms count (the rest are reported as dropped), and
    every task has the same weight, so a different task mix in the two arms cannot create or hide
    a difference. Interval: Student t on the per-task differences, n_tasks − 1 degrees of freedom;
    90% two-sided, so each end is a one-sided 95% bound. With fewer than 2 tasks there is no interval."""
    per = collections.defaultdict(lambda: {arm: [0, 0], base: [0, 0]})
    for r in rows:
        if r["condition"] in (arm, base) and answered(r):
            c = per[r["task_id"]][r["condition"]]
            c[0] += grading.attempted_hack(r)
            c[1] += 1
    both = sorted(t for t, c in per.items() if c[arm][1] and c[base][1])
    d = np.array([per[t][arm][0] / per[t][arm][1] - per[t][base][0] / per[t][base][1] for t in both])
    out = {"n_tasks": len(both), "n_tasks_dropped": len(per) - len(both), "diff": math.nan, "lo": math.nan, "hi": math.nan}
    if len(d):
        out["diff"] = float(d.mean())
    if len(d) >= 2:
        half = float(sps.t.ppf(1 - (1 - conf) / 2, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))
        out.update(lo=out["diff"] - half, hi=out["diff"] + half)
    return out


def verdict(c: dict, margin_pp: float) -> str:
    """The preregistered reading of one contrast against the margin (in percentage points)."""
    if math.isnan(c["lo"]):
        return "no interval (fewer than 2 tasks with both arms)"
    m = margin_pp / 100
    if c["hi"] < -m:
        return f"lowers the attempt rate by more than {margin_pp:g} pp"
    if c["lo"] > m:
        return f"raises the attempt rate by more than {margin_pp:g} pp"
    if -m < c["lo"] and c["hi"] < m:
        return f"changes the attempt rate by less than {margin_pp:g} pp"
    return "inconclusive at this margin"


def analyze(cfg: dict, pilot: bool) -> None:
    pre = "pilot:" if pilot else ""
    for side, cell_key in SIDES.items():
        cell = cfg["cells"][cell_key]
        rows = [dict(r, condition=r["condition"][len(pre):]) for r in store.RolloutStore(cell["arm_store"]).rows()
                if r["condition"].startswith("pilot:") == pilot]
        print(f"\n{_ckpt(cell).label}: attempted-hack rate among answered samples")
        for arm, t in arm_table(rows).items():
            rate = f"{100 * t['rate']:5.1f}% [{100 * t['lo']:.1f}, {100 * t['hi']:.1f}]" if t["answered"] else "n/a"
            print(f"  {arm:14} {rate:26} {t['attempted']}/{t['answered']} answered; {t['truncated']} truncated, "
                  f"{t['no_code']} no code, {t['reopened_think']} reopened reasoning; {t['hacked']} hacked; {t['n_tasks']} tasks")
        present = {r["condition"] for r in rows}
        for arm, base in cfg["contrasts"][side].items():
            if arm in present and base in present:
                c = contrast(rows, arm, base)
                star = " [primary]" if [arm, base] == cfg["primary"].get(side) else ""
                print(f"  {arm} − {base}: {100 * c['diff']:+.1f} pp, 90% interval [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}], "
                      f"{c['n_tasks']} tasks ({c['n_tasks_dropped']} dropped): {verdict(c, cfg['decision']['margin_pp'])}{star}")


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
        jobs += [(cfg["cells"][cell], cfg["cells"][cell]["arm_store"], b[side]) for side, cell in SIDES.items()]
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
