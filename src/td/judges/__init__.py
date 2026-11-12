"""LLM judges over OpenRouter: monitors (upstream), silence judges, admission scorer.

All calls go through `budgeted_call`, which reuses upstream's `scripts.monitor_judge.call`
(temperature 0, provider reasoning off) and upstream's retry policy (up to 6 attempts; attempts
after the first at temperature 0.7, used when a reply does not parse), and reserves the worst case
against the milestone budget before each attempt.

Judges call it through `judged`, which adds the cache lookup and writes the reply to the
`JudgementStore` as soon as it arrives, so a paid reply is never lost or bought twice.
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Callable

from td import config, cost, store

MAX_ATTEMPTS = 6
MAX_OUT_TOKENS = 8000  # upstream monitor_judge.call


async def budgeted_call(model: str, system: str | None, user: str, parse: Callable[[str], object],
                        budget: cost.Budget, what: str) -> dict:
    """Call `model` until `parse(text)` returns a non-None value or attempts run out.
    Returns {text, parsed, attempts, itok, otok, ms, cost_usd, error}. Never logs keys.

    `parse` must return None for a reply it can't read; that is what triggers a retry. Upstream's
    monitor parser returns `(None, None)` instead, so wrap it with `td.judges.monitors.parser`.
    On failure, `text` is the last reply received (None if every attempt raised), kept for debugging."""
    from scripts.monitor_judge import call

    config.require_key("OPENROUTER_API_KEY")
    # Upper bound on input tokens: every token is at least one byte.
    n_in_bound = len(((system or "") + user).encode())
    total, last_err, last = 0.0, None, {"text": None, "itok": None, "otok": None, "ms": None}
    for attempt in range(MAX_ATTEMPTS):
        est = cost.openrouter_cost(model, n_in_bound, MAX_OUT_TOKENS)
        with budget.reserve(est, f"{what} attempt {attempt}") as r:
            try:
                text, ms, (itok, otok) = await asyncio.to_thread(call, model, system, user, 0 if attempt == 0 else 0.7)
            except Exception as e:  # noqa: BLE001 — network / rate limit: back off and retry
                last_err = f"{type(e).__name__}"  # never include the message: it may echo headers
                r.release("call failed")  # assume unbilled; reconcile against OpenRouter billing
                await asyncio.sleep(2 ** min(attempt, 4) * 5 + random.random() * 5)
                continue
            usd = cost.openrouter_cost(model, itok or n_in_bound, otok or MAX_OUT_TOKENS)
            r.settle(usd, {"provider": "openrouter", "model": model, "itok": itok, "otok": otok})
            total += usd
        last, last_err = {"text": text, "itok": itok, "otok": otok, "ms": ms}, None
        parsed = parse(text)
        if parsed is not None:
            return {**last, "parsed": parsed, "attempts": attempt + 1, "cost_usd": total, "error": None}
    return {**last, "parsed": None, "attempts": MAX_ATTEMPTS, "cost_usd": total,
            "error": last_err or "unparseable"}


async def judged(js: store.JudgementStore, *, rollout_id: str, judge: str, model: str, prompt_version: str,
                 system: str | None, user: str, parse: Callable[[str], object], budget: cost.Budget,
                 what: str) -> dict:
    """One judgement, as a complete `JudgementStore` row.

    A row already stored under (rollout_id, judge, prompt_version, model) is returned without a
    call. Otherwise the judge is called and a parsed reply is appended to `js` at once, before the
    caller sees it. A reply that never parsed is returned but not stored, so a rerun retries it
    (upstream does the same); its cost is in the ledger either way."""
    jid = store.judgement_key(rollout_id, judge, prompt_version, model)
    hit = js.get(jid)
    if hit is not None:
        return hit
    out = await budgeted_call(model, system, user, parse, budget, what)
    row = {"rollout_id": rollout_id, "judge": judge, "model": model, "prompt_version": prompt_version,
           "output": out, "created_at": store.now_iso(), "cost_usd": out["cost_usd"]}
    if out["parsed"] is None:
        return row
    js.append(row)
    return js.get(jid)
