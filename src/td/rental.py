"""A machine rented by the hour, in the spending ledger.

Per-token providers reserve and settle one sample at a time. A rented machine is one bill, so it
is one reservation: opened for the most we are willing to spend before the machine starts, and
settled with the provider's actual charge when the machine is destroyed. While it is open it
counts against the milestone's cap like any other reservation.

    bin/td python -m td.rental open  M5a 75 "vast 4xA100, step 2"        -> prints the reservation id
    bin/td python -m td.rental close M5a <rid> 43.20 --note "vast invoice, 9.5 h"
    bin/td python -m td.rental status M5a
"""
from __future__ import annotations

import argparse

from td import cost


def open_rental(budget: cost.Budget, usd: float, what: str) -> str:
    """Reserve `usd` for a rental. Raises `CapExceeded` if it does not fit under the cap."""
    return budget.reserve(usd, f"rental: {what}").rid


def close_rental(budget: cost.Budget, rid: str, actual_usd: float, note: str = "") -> None:
    """Settle an open rental with what the provider charged. Refuses an unknown or closed id, and
    refuses a charge above the reservation (that needs the user's OK and a new reservation)."""
    with cost._locked(budget.ledger) as f:
        events = [e for e in cost._read(f) if e.get("milestone") == budget.milestone]
    opened = {e["rid"]: e for e in events if e["type"] == "reserve"}
    closed = {e["rid"] for e in events if e["type"] in ("settle", "release")}
    if rid not in opened or rid in closed:
        raise RuntimeError(f"{rid} is not an open reservation of {budget.milestone}")
    e = opened[rid]
    if not e["what"].startswith("rental: "):
        raise RuntimeError(f"{rid} is not a rental ({e['what']!r})")
    cost.Reservation(budget, rid, e["usd"], e["what"]).settle(actual_usd, {"provider": "rental", "note": note})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("open"); p.add_argument("milestone"); p.add_argument("usd", type=float); p.add_argument("what")
    p = sub.add_parser("close"); p.add_argument("milestone"); p.add_argument("rid"); p.add_argument("usd", type=float)
    p.add_argument("--note", default="")
    p = sub.add_parser("status"); p.add_argument("milestone")
    a = ap.parse_args()
    b = cost.Budget(a.milestone)
    if a.cmd == "open":
        print(open_rental(b, a.usd, a.what))
    elif a.cmd == "close":
        close_rental(b, a.rid, a.usd, a.note)
    s = b.status()
    print(f"{a.milestone}: cap ${s['cap']:.2f}, settled ${s['settled']:.2f}, outstanding ${s['outstanding']:.2f} "
          f"in {s['n_open']} open, remaining ${s['remaining']:.2f}")
