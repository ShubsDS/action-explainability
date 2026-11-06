"""Spending caps, enforced in code (handoff rule 1).

Every paid job goes through a `Budget` for its milestone:

    budget = Budget("M1")
    est = tinker_rollout_worst_case(n_prompt_tokens)
    with budget.reserve(est, what="m1 anti_s1@30 task X sample 3") as r:
        result = sample(...)
        r.settle(tinker_cost(n_prompt, n_output))

`reserve` refuses (raises `CapExceeded`) if outstanding reservations + settled spend + the new
worst case would pass the milestone's cap. A reservation that is never settled (crash) stays
counted against the cap, which errs on the safe side; `release_stale` clears it after a manual check.

The ledger (`costs/ledger.jsonl`) is append-only and committed. Events:
  reserve   {rid, milestone, usd, what}
  settle    {rid, milestone, usd (actual), usage}
  release   {rid, milestone, reason}         (reservation cancelled, nothing spent)
  reconcile {milestone, provider, billed_usd, ledger_usd, note}
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from td import config


class CapExceeded(RuntimeError):
    pass


class KillRuleBlocked(RuntimeError):
    pass


# ---------------------------------------------------------------- estimates (USD)

def tinker_cost(n_prompt: int, n_output: int, n_cached_prompt: int = 0, prices: dict | None = None) -> float:
    p = prices or config.PRICES["tinker"]
    uncached = max(0, n_prompt - n_cached_prompt)
    return (n_output * p["sample_output"] + uncached * p["prefill"]
            + n_cached_prompt * p["prefill_cached"]) / 1e6


def tinker_rollout_worst_case(n_prompt: int, max_tokens: int = config.MAX_TOKENS) -> float:
    """Every rollout at its full output budget, prompt prefilled uncached."""
    return tinker_cost(n_prompt, max_tokens)


def openrouter_cost(model: str, n_in: int, n_out: int) -> float:
    try:
        p = config.PRICES["openrouter"][model]
    except KeyError:
        raise KeyError(f"no OpenRouter price for {model!r} in td.config.PRICES['openrouter']; "
                       "look it up and record it (and in NOTES.md) before reserving") from None
    return (n_in * p["input"] + n_out * p["output"]) / 1e6


def judge_worst_case(model: str, n_in: int, max_out: int = 8000, attempts: int = 6) -> float:
    """Upstream monitor_judge retries up to 6 times with max_tokens 8000; reserve for all of them."""
    return attempts * openrouter_cost(model, n_in, max_out)


# ---------------------------------------------------------------- ledger

_THREAD_LOCK = threading.Lock()


@contextlib.contextmanager
def _locked(path: Path):
    """Process-safe (flock) and thread-safe exclusive access to the ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCK, open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield f
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read(f) -> list[dict]:
    f.seek(0)
    return [json.loads(line) for line in f if line.strip()]


def _append(f, event: dict) -> None:
    f.seek(0, 2)
    f.write(json.dumps({"ts": time.time(), **event}) + "\n")
    f.flush()


def summarize(events: list[dict], milestone: str) -> dict:
    """{'settled': $, 'outstanding': $, 'n_open': int} for one milestone."""
    reserved: dict[str, float] = {}
    settled = 0.0
    for e in events:
        if e.get("milestone") != milestone:
            continue
        t = e["type"]
        if t == "reserve":
            reserved[e["rid"]] = e["usd"]
        elif t == "settle":
            reserved.pop(e["rid"], None)
            settled += e["usd"]
        elif t == "release":
            reserved.pop(e["rid"], None)
    return {"settled": settled, "outstanding": sum(reserved.values()), "n_open": len(reserved),
            "open_rids": sorted(reserved)}


@dataclass
class Reservation:
    budget: "Budget"
    rid: str
    usd: float
    what: str
    done: bool = field(default=False)

    def settle(self, actual_usd: float, usage: dict | None = None) -> None:
        if self.done:
            raise RuntimeError(f"reservation {self.rid} already closed")
        if actual_usd > self.usd + 1e-9:
            # Never silently exceed a worst case: it means the estimate was wrong.
            raise RuntimeError(f"actual cost ${actual_usd:.6f} > reserved worst case ${self.usd:.6f} "
                               f"for {self.what}; fix the estimate")
        self.budget._event({"type": "settle", "rid": self.rid, "usd": actual_usd, "usage": usage or {}})
        self.done = True

    def release(self, reason: str) -> None:
        if not self.done:
            self.budget._event({"type": "release", "rid": self.rid, "reason": reason})
            self.done = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Unsettled on a clean exit = nothing was billed (e.g. cache hit). On an exception we keep
        # the reservation open: the call may have been billed even though we never saw a result.
        if not self.done and exc_type is None:
            self.release("closed without settle")
        return False


class Budget:
    def __init__(self, milestone: str, ledger: Path | None = None, caps: dict | None = None,
                 kill_rule_settled: bool | None = None):
        caps = caps if caps is not None else config.CAPS
        if milestone not in caps:
            raise KeyError(f"unknown milestone {milestone!r}; caps: {sorted(caps)}")
        self.milestone = milestone
        self.cap = float(caps[milestone])
        self.ledger = ledger or config.LEDGER_PATH
        self._kill_ok = config.KILL_RULE_SETTLED if kill_rule_settled is None else kill_rule_settled

    def _event(self, e: dict) -> None:
        with _locked(self.ledger) as f:
            _append(f, {"milestone": self.milestone, **e})

    def status(self) -> dict:
        with _locked(self.ledger) as f:
            s = summarize(_read(f), self.milestone)
        s["cap"] = self.cap
        s["remaining"] = self.cap - s["settled"] - s["outstanding"]
        return s

    def reserve(self, usd: float, what: str) -> Reservation:
        if usd < 0:
            raise ValueError("negative reservation")
        if self.milestone in config.MILESTONES_BLOCKED_BY_KILL_RULE and not self._kill_ok:
            raise KillRuleBlocked(f"{self.milestone} paid work is blocked until the user says the kill-rule "
                                  "question (handoff rule 2) is settled; see td.config.KILL_RULE_SETTLED")
        with _locked(self.ledger) as f:
            s = summarize(_read(f), self.milestone)
            committed = s["settled"] + s["outstanding"]
            if committed + usd > self.cap + 1e-9:
                raise CapExceeded(f"{self.milestone}: reserving ${usd:.4f} for {what!r} would bring committed "
                                  f"spend to ${committed + usd:.4f} > cap ${self.cap:.2f} "
                                  f"(settled ${s['settled']:.4f}, outstanding ${s['outstanding']:.4f}). "
                                  "Raising a cap needs the user's OK.")
            rid = uuid.uuid4().hex
            _append(f, {"milestone": self.milestone, "type": "reserve", "rid": rid, "usd": usd, "what": what})
        return Reservation(self, rid, usd, what)

    def release_stale(self, rids: list[str], reason: str) -> None:
        """Release reservations left open by a crash, after checking provider billing by hand."""
        for rid in rids:
            self._event({"type": "release", "rid": rid, "reason": f"stale: {reason}"})

    def reconcile(self, provider: str, billed_usd: float, note: str = "") -> dict:
        """Record a billing-vs-ledger check (end of every milestone)."""
        s = self.status()
        self._event({"type": "reconcile", "provider": provider, "billed_usd": billed_usd,
                     "ledger_usd": s["settled"], "note": note})
        return {**s, "billed_usd": billed_usd, "diff_usd": billed_usd - s["settled"]}
