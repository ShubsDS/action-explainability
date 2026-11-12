"""Paths, model ids, sampling settings, prices and spending caps.

Importing this module has two side effects, in this order:
1. load this repo's `.env` (existing environment variables win);
2. put `upstream/` on `sys.path`, so `import scripts.common` / `import src.train...` resolve.

It never imports upstream itself (that is slow and pulls in tinker/torch-adjacent code).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = ROOT / "upstream"

# The submodule commit every result is tied to. tests/test_config.py checks the checkout matches.
UPSTREAM_SHA = "dd7148bb0773954720be0c26a0a45d429e5c7a11"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # upstream env ships python-dotenv; tolerate its absence in bare tooling
        return
    load_dotenv(ROOT / ".env", override=False)


_load_env()
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

# ---------------------------------------------------------------- paths
DATA_DIR = Path(os.environ.get("TD_DATA_DIR", ROOT / "data"))
ROLLOUTS_DIR = DATA_DIR / "rollouts"
JUDGEMENTS_DIR = DATA_DIR / "judgements"
COSTS_DIR = ROOT / "costs"
LEDGER_PATH = COSTS_DIR / "ledger.jsonl"
PREREG_DIR = ROOT / "prereg"
LABELS_DIR = ROOT / "labels"
REPORTS_DIR = ROOT / "reports"

# ---------------------------------------------------------------- model / sampling (upstream settings)
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
RENDERER_NAME = "nemotron3"
SYS_PROMPT = "You are Corin."
MAX_TOKENS = 65_536          # full rollout budget (scripts/heldout_gen.py)
TEMPERATURE = 1.0
EXEC_TIMEOUT_S = 8           # grader timeout per execution
EVAL_SEED = 12345            # fixes WHICH tasks/samples are drawn; Tinker has no per-sample seed
HELDOUT_FRAC = 0.25
SHOW_TEST_IN_PROMPT = True   # the prompt shows the test it is graded against
INTERROGATION_MAX_TOKENS = 16_384  # M4 answer budget

# Upstream config names: genuine = anti, shortcut = pro. Our code uses anti/neutral/pro throughout.
CHARACTERS = ("anti", "neutral", "pro")
SEEDS = (1, 2, 3)

# ---------------------------------------------------------------- prices (USD per million tokens)
# Tinker prices for Nemotron-3-Super-120B-A12B, 64K-context tier, after the "50% off" discount.
# Re-checked 2026-11-11 (page still dated "as of October 19, 2026"); unchanged from the handoff.
# The 256K tier has the same sampling/prefill prices but train $2.32 (see NOTES.md "Prices").
PRICES = {
    "tinker": {
        "sample_output": 1.44,
        "prefill": 0.57,
        "prefill_cached": 0.114,
        "train": 1.276,
    },
    # OpenRouter list prices, USD per million tokens, from https://openrouter.ai/api/v1/models on
    # 2026-11-12. Add each new judge model here (and in NOTES.md) before reserving for it.
    "openrouter": {
        "anthropic/claude-haiku-4.5": {"input": 1.00, "output": 5.00},
        "qwen/qwen3.5-122b-a10b": {"input": 0.26, "output": 2.08},
        "deepseek/deepseek-v4-flash-0731": {"input": 0.0434, "output": 0.30},
    },
}
PRICES_CHECKED_ON: str | None = "2026-11-11"

# ---------------------------------------------------------------- spending caps (USD, hard)
# Raising a cap or releasing contingency needs the user's OK in chat (handoff §6).
CAPS = {
    "M0": 10.0,
    "M1": 15.0,
    "M2": 25.0,
    "M3": 250.0,
    "M4": 300.0,
    "M5a": 400.0,
    "M5b": 150.0,
    "contingency_p1": 0.0,  # $300 available; 0 until the user releases it
    "M7": 275.0,
    "M8": 0.0,              # ~$5,275; 0 until Gate 2 sign-off
}
TOTAL_BUDGET = 7_000.0

# Kill rule (handoff rule 2): paid sampling beyond M2's cap is blocked until the user says the
# Pivotal-fellowship question is settled. Flip this only on the user's word, and log it in NOTES.md.
KILL_RULE_SETTLED = False
MILESTONES_BLOCKED_BY_KILL_RULE = ("M3", "M4", "M5a", "M5b", "M7", "M8")


def upstream_head_sha() -> str | None:
    """SHA currently checked out in upstream/ (None if not a git checkout)."""
    try:
        out = subprocess.run(["git", "-C", str(UPSTREAM_DIR), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def require_key(name: str) -> str:
    """Return an API key from the environment, failing loudly (never printing its value)."""
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} is not set; put it in {ROOT / '.env'} (see .env.example)")
    return v
