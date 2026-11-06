"""Intervals and tests used across milestones.

- `wilson`: single-cell binomial counts (e.g. M1's 22×5 cell).
- `cluster_bootstrap`: rates over many tasks; resamples TASKS with replacement (>= 2,000 draws).
- `one_sided_upper`: e.g. M3's false-silent rate upper bound.
- `tost_diff`: M5 equivalence (±5 pp) on a paired / clustered difference; filled in for M5.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats

N_BOOT = 2000


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    z = float(stats.norm.ppf(1 - (1 - conf) / 2))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_upper_one_sided(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided upper bound (a two-sided interval at 1 - 2(1 - conf))."""
    return wilson(k, n, conf=1 - 2 * (1 - conf))[1]


def cluster_bootstrap(values: Sequence[float], clusters: Sequence[str], *,
                      stat: Callable[[np.ndarray], float] = np.mean, n_boot: int = N_BOOT,
                      seed: int = 0, conf: float = 0.95) -> dict:
    """Task-clustered percentile bootstrap of `stat` over pooled observations.
    Returns {est, lo, hi, n, n_clusters, draws}."""
    if n_boot < N_BOOT:
        raise ValueError(f"use at least {N_BOOT} draws (handoff §7)")
    by = defaultdict(list)
    for v, c in zip(values, clusters, strict=True):
        by[c].append(v)
    keys = sorted(by)
    groups = [np.asarray(by[k], dtype=float) for k in keys]
    rng = np.random.default_rng(seed)
    est = float(stat(np.concatenate(groups))) if groups else math.nan
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        draws[b] = stat(np.concatenate([groups[i] for i in idx]))
    a = (1 - conf) / 2
    return {"est": est, "lo": float(np.quantile(draws, a)), "hi": float(np.quantile(draws, 1 - a)),
            "n": int(sum(len(g) for g in groups)), "n_clusters": len(groups), "draws": draws}


def cluster_bootstrap_one_sided(values, clusters, *, upper: bool, conf: float = 0.95, **kw) -> float:
    r = cluster_bootstrap(values, clusters, conf=1 - 2 * (1 - conf), **kw)
    return r["hi"] if upper else r["lo"]


def fmt_rate(k: int, n: int, lo: float, hi: float) -> str:
    """'54.5% (60/110; 95% CI 45.2–63.5%)': n beside every rate."""
    return f"{100 * k / n:.1f}% ({k}/{n}; 95% CI {100 * lo:.1f}–{100 * hi:.1f}%)" if n else "n/a (n=0)"


def tost_diff(*args, **kwargs):
    """Two one-sided tests for |arm − P| < 5 pp, on the M5 mixed model's contrast. Built in M5,
    once prereg/m5.yaml freezes the model and margin."""
    raise NotImplementedError("M5: TOST equivalence on the mixed-model contrast")
