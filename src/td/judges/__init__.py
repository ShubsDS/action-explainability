"""LLM judges over OpenRouter: monitors (upstream), silence judges, admission scorer.

All calls go through `budgeted_call`, which reuses upstream's `scripts.monitor_judge.call`
(temperature 0, provider reasoning off) and upstream's retry policy (up to 6 attempts; attempts
after the first at temperature 0.7, used when a reply does not parse), and reserves the worst case
against the milestone budget before each attempt.
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Callable

from td import config, cost

MAX_ATTEMPTS = 6
MAX_OUT_TOKENS = 8000  # upstream monitor_judge.call


async def budgeted_call(model: str, system: str | None, user: str, parse: Callable[[str], object],
                        budget: cost.Budget, what: str) -> dict:
    """Call `model` until `parse(text)` returns a non-None value or attempts run out.
    Returns {text, parsed, attempts, itok, otok, ms, cost_usd, error}. Never logs keys."""
    from scripts.monitor_judge import call

    config.require_key("OPENROUTER_API_KEY")
    # Upper bound on input tokens: every token is at least one byte.
    n_in_bound = len(((system or "") + user).encode())
    total, last_err = 0.0, None
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
        parsed = parse(text)
        if parsed is not None:
            return {"text": text, "parsed": parsed, "attempts": attempt + 1, "itok": itok, "otok": otok,
                    "ms": ms, "cost_usd": total, "error": None}
    return {"text": None, "parsed": None, "attempts": MAX_ATTEMPTS, "itok": None, "otok": None, "ms": None,
            "cost_usd": total, "error": last_err or "unparseable"}
