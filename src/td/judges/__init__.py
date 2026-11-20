"""LLM judges over OpenRouter: monitors (upstream), silence judges, admission scorer.

All calls go through `budgeted_call`, which reuses upstream's `scripts.monitor_judge.call`
(temperature 0, provider reasoning off; `reasoning=True` uses `call_with_reasoning` instead) and upstream's retry policy (up to 6 attempts; attempts
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


def call_with_reasoning(model: str, system: str | None, user: str, temp: float = 0):
    """Upstream's `call` with the provider's reasoning left ON, returning the visible reply only.

    Upstream switches reasoning off, which some endpoints refuse (HTTP 400, "Reasoning is mandatory",
    seen on Google models 2026-11-20) and which makes a small model skim a long text. Upstream also
    appends the reasoning channel to the reply; here it is dropped, so tags are read from the answer
    alone. Same request otherwise: max_tokens 8000 (reasoning included), same endpoint and timeout."""
    import json
    import os
    import time
    import urllib.request

    t0 = time.monotonic()
    body = {"model": model, "max_tokens": MAX_OUT_TOKENS, "temperature": temp, "reasoning": {"enabled": True},
            "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        r = json.loads(resp.read())
    u = r.get("usage") or {}
    return (r["choices"][0]["message"].get("content") or "", int((time.monotonic() - t0) * 1000),
            (u.get("prompt_tokens"), u.get("completion_tokens")))


async def budgeted_call(model: str, system: str | None, user: str, parse: Callable[[str], object],
                        budget: cost.Budget, what: str, reasoning: bool = False) -> dict:
    """Call `model` until `parse(text)` returns a non-None value or attempts run out.
    Returns {text, parsed, attempts, itok, otok, ms, cost_usd, error}. Never logs keys.

    `parse` must return None for a reply it can't read; that is what triggers a retry. Upstream's
    monitor parser returns `(None, None)` instead, so wrap it with `td.judges.monitors.parser`.
    On failure, `text` is the last reply received (None if every attempt raised), kept for debugging."""
    from scripts import monitor_judge

    call = call_with_reasoning if reasoning else monitor_judge.call   # looked up per call, so tests can replace either
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
                 what: str, reasoning: bool = False) -> dict:
    """One judgement, as a complete `JudgementStore` row.

    A row already stored under (rollout_id, judge, prompt_version, model) is returned without a
    call. Otherwise the judge is called and a parsed reply is appended to `js` at once, before the
    caller sees it. A reply that never parsed is returned but not stored, so a rerun retries it
    (upstream does the same); its cost is in the ledger either way."""
    jid = store.judgement_key(rollout_id, judge, prompt_version, model)
    hit = js.get(jid)
    if hit is not None:
        return hit
    out = await budgeted_call(model, system, user, parse, budget, what, reasoning=reasoning)
    row = {"rollout_id": rollout_id, "judge": judge, "model": model, "prompt_version": prompt_version,
           "output": out, "created_at": store.now_iso(), "cost_usd": out["cost_usd"]}
    if out["parsed"] is None:
        return row
    js.append(row)
    return js.get(jid)
