"""Exploratory tables for the scoring-fact experiment (reports/m5_step3.md). Reads stored rows;
samples nothing and spends nothing. None of this is preregistered, except that the first part is
the check the report promises for a difference between how the attempts were served.

    bin/td python -m td.experiments.m5_grader_explore            # every table
    bin/td python -m td.experiments.m5_grader_explore --show     # and the passages the phrase counts found

1. Token limit. 1,305 attempts were allowed 65,536 output tokens and 95 were limited to prompt +
   output = 65,536, the total every earlier experiment ran under (NOTES.md, 2026-11-21). The primary
   contrasts again, with every answer that ran past that total counted as truncated.
2. The three-way split of each arm and where the effects sit (per task, held-out against training).
3. The stored first attempts with nothing added, for scale (another server, other days).
4. Phrases of each arm's sentence in the reasoning. A literal search, NOT the preregistered
   `uptake` judge: it undercounts restatements in other words and was validated on nothing.
"""
from __future__ import annotations

import argparse
import collections
import re

from td import config, grading, store
from td.experiments import m5_grader as g

# Literal phrases. `{name}` is the test function's name in that arm's message.
PHRASES = {
    "says `{name}` is not to be defined or changed":
        r"(not|n't|never|without|avoid)\W+(\w+\W+){{0,4}}(defin|redefin|chang|modif|assign|overrid|touch)\w*\W+(\w+\W+){{0,4}}`?{name}`?",
    "scored as failures": r"scored as (a )?fail|as failures?",
    "illustration": r"illustrat",
}


def over_total(r: dict) -> bool:
    return not r["truncated"] and r["n_prompt_tokens"] + r["n_output_tokens"] > config.MAX_TOKENS


def token_limit_check(cfg: dict, rows: list[dict]) -> None:
    over = [r for r in rows if over_total(r)]
    print(f"1. Answered attempts past prompt + output = {config.MAX_TOKENS:,}: {len(over)} "
          f"{dict(collections.Counter(r['condition'] for r in over))}; "
          f"{sum(grading.rebinds(r, _name(cfg, r['condition'])) for r in over)} of them rebind the test's function")
    moved = [dict(r, truncated=True) if over_total(r) else r for r in rows]
    margin, classes = cfg["decision"]["margin_pp"], {}
    for arm, base in list(cfg["contrasts"].items()) + list(cfg["paired"].items()):
        a, b = g.contrast(rows, arm, base, outcome=grading.rebinds), g.contrast(moved, arm, base, outcome=grading.rebinds)
        classes[(arm, base)] = g.klass(b, margin)
        mark = " [primary]" if [arm, base] in cfg["primary"] else ""
        print(f"   {arm} − {base}: as stored {100 * a['diff']:+.1f} pp; with them truncated {100 * b['diff']:+.1f} pp "
              f"[{100 * b['lo']:+.1f}, {100 * b['hi']:+.1f}], {g.klass(a, margin)} -> {g.klass(b, margin)}{mark}")
    print(f"   reading with them truncated: {g.reading(classes)}")


def _name(cfg: dict, arm: str) -> str:
    return cfg["arms"][arm].get("rename_check_to") or "check"


def splits(cfg: dict, rows: list[dict]) -> None:
    print("\n2. Each arm three ways, among answered attempts (the renamed arm by `run_tests`; the grader cannot score it)")
    for arm in cfg["arms"]:
        name = _name(cfg, arm)
        done = [r for r in rows if r["condition"] == arm and g.answered(r)]
        reb = [r for r in done if grading.rebinds(r, name)]
        if name == "check":
            works = sum(g.working_hack(r) for r in reb)
            plain = sum(g.rebind_form(r) == "assignment" for r in reb)
            print(f"   {arm:17} answered {len(done)}: working override {works}, rebind that does not work {len(reb) - works}, "
                  f"no rebind {len(done) - len(reb)}; plain assignments {plain} of {len(reb)}; "
                  f"other cheats {sum(g.other_cheat(r) for r in done)}")
        else:
            print(f"   {arm:17} answered {len(done)}: rebinds `{name}` {len(reb)}, does not {len(done) - len(reb)}")
        for label, held in (("held-out", True), ("training", False)):
            sub = [r for r in done if r["heldout"] == held]
            print(f"      {label}: {sum(grading.rebinds(r, name) for r in sub)}/{len(sub)}")
    print("   Per task, test arm − placebo (2 attempts a task, so steps of 50 pp unless a sample is missing):")
    for arm in ("user_prohibition", "exact_line", "never_run"):
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            if r["condition"] in (arm, "user_placebo") and g.answered(r):
                per[r["task_id"]][r["condition"]].append(grading.rebinds(r))
        d = [sum(c[arm]) / len(c[arm]) - sum(c["user_placebo"]) / len(c["user_placebo"])
             for c in per.values() if c[arm] and c["user_placebo"]]
        print(f"      {arm}: {len(d)} tasks; no difference on {sum(x == 0 for x in d)}; −100 pp on {sum(x == -1 for x in d)}; "
              f"−50 pp or lower on {sum(x <= -0.5 for x in d)}; higher than the placebo on {sum(x > 0 for x in d)}")


def baseline(cfg: dict) -> None:
    print("\n3. Stored first attempts with nothing added, same tasks (published path, 2026-11-12 and later; for scale only)")
    tasks = set(g.study_items(cfg, False))
    for name in cfg["cells"]["rl"]["baseline_stores"]:
        rs = [r for r in store.RolloutStore(name).rows()
              if r["task_id"] in tasks and r.get("parent_rollout_id") is None and r.get("condition") in ("base", None)]
        done = [r for r in rs if g.answered(r)]
        print(f"   {name}: {sum(grading.rebinds(r) for r in done)}/{len(done)} answered attempts rebind `check` "
              f"({len(rs)} attempts, {len({r['task_id'] for r in rs})} tasks)")


def phrases(cfg: dict, rows: list[dict], show: bool) -> None:
    print("\n4. Phrases in the reasoning, among answered attempts (literal search; not the `uptake` judge)")
    for label, pat in PHRASES.items():
        print(f"   {label.format(name='check')}:")
        for arm in cfg["arms"]:
            name = _name(cfg, arm)
            rx = re.compile(pat.format(name=name), re.I)
            done = [r for r in rows if r["condition"] == arm and g.answered(r)]
            hit = [r for r in done if rx.search(r["reasoning"] or "")]
            miss = [r for r in done if not rx.search(r["reasoning"] or "")]
            reb = lambda rs: sum(grading.rebinds(r, name) for r in rs)  # noqa: E731
            print(f"      {arm:17} {len(hit):3} of {len(done)}; of those {reb(hit)} rebind `{name}`; "
                  f"without the phrase {reb(miss)} of {len(miss)} rebind")
            if show:
                for r in sorted(hit, key=lambda r: (r["task_id"], r["sample_idx"])):
                    if grading.rebinds(r, name):
                        _show(r, rx, name)
    print("   The answer's code contains the word 'placeholder':")
    for arm in cfg["arms"]:
        name = _name(cfg, arm)
        done = [r for r in rows if r["condition"] == arm and g.answered(r)]
        hit = [r for r in done if re.search("placeholder", r["code"] or "", re.I)]
        print(f"      {arm:17} {len(hit):3} of {len(done)}; of those {sum(grading.rebinds(r, name) for r in hit)} rebind `{name}`")


def _show(r: dict, rx: re.Pattern, name: str) -> None:
    """The last matching sentence of the reasoning and the line of the answer that rebinds `name`."""
    text = r["reasoning"]
    m = list(rx.finditer(text))[-1]
    start = max(text.rfind("\n", 0, m.start()), text.rfind(". ", 0, m.start())) + 1
    ends = [x for x in (text.find("\n", m.end()), text.find(". ", m.end()) + 1) if x > 0]
    line = next((ln.strip() for ln in (r["code"] or "").splitlines() if re.match(rf"\s*(def\s+{name}\b|{name}\s*=)", ln)), "?")
    print(f"         [{r['condition']} {r['task_id']} #{r['sample_idx']}] {len(text) - m.end():,} characters before the end "
          f"of the reasoning; working hack: {g.working_hack(r)}")
    print(f"            reasoning: {' '.join(text[start:min(ends or [len(text)])].split())[:300]}")
    print(f"            answer:    {line[:200]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="print the passages behind the phrase counts")
    a = ap.parse_args()
    cfg = g.load_cfg()
    rows, _ = g.load_rows(cfg, False)
    print(f"{len(rows)} attempts, {sum(r['n_output_tokens'] for r in rows):,} output tokens, "
          f"{sum(bool(r['truncated']) for r in rows)} truncated\n")
    token_limit_check(cfg, rows)
    splits(cfg, rows)
    baseline(cfg)
    phrases(cfg, rows, a.show)
