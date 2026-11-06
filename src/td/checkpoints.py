"""The 30 released checkpoints, read through upstream's own table (configs/checkpoints.json).

Identify a checkpoint by (character, seed, step), never by Tinker run id (that changes
along a run). Step 0 is the character's SFT checkpoint, shared by all three seeds, so its
`seed` is None.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from td import config  # noqa: F401  (loads .env, puts upstream/ on sys.path)

import scripts.common as _up  # upstream: CHECKPOINTS, tinker_path, crossing_step

CHARACTERS = tuple(_up.CHARACTERS)  # ("pro", "neutral", "anti")
SEEDS = (1, 2, 3)

if _up.BASE_MODEL != config.BASE_MODEL:
    raise RuntimeError(f"upstream base model {_up.BASE_MODEL!r} != td.config {config.BASE_MODEL!r}")


@dataclass(frozen=True)
class Checkpoint:
    character: str
    seed: int | None      # None for the SFT checkpoint (step 0)
    step: int
    sampler_path: str     # tinker://.../sampler_weights/...
    state_path: str | None  # tinker://.../weights/... — NOT in upstream's table @ pinned SHA (NOTES.md)
    hf_repo: str

    @property
    def run(self) -> str | None:
        """Upstream run key, e.g. 'anti_s1' (None for SFT)."""
        return None if self.seed is None else f"{self.character}_s{self.seed}"

    @property
    def label(self) -> str:
        return f"{self.character}-sft" if self.step == 0 else f"{self.run}@{self.step}"

    def key(self) -> dict:
        """The part of a checkpoint that goes into cache keys and rollout records."""
        return {"character": self.character, "seed": self.seed, "step": self.step,
                "sampler_path": self.sampler_path}

    def as_dict(self) -> dict:
        return asdict(self)


def _check_char(character: str) -> None:
    if character not in CHARACTERS:
        raise ValueError(f"unknown character {character!r}; use one of {CHARACTERS} "
                         "(upstream configs call anti 'genuine' and pro 'shortcut')")


def sft(character: str) -> Checkpoint:
    _check_char(character)
    e = _up.CHECKPOINTS["sft"][character]
    return Checkpoint(character, None, 0, e["tinker"], None, e["hf"])


def get(character: str, seed: int | None, step: int) -> Checkpoint:
    """Released checkpoint at (character, seed, step). Step 0 → SFT (seed ignored)."""
    _check_char(character)
    if step == 0:
        return sft(character)
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS} for RL steps, got {seed!r}")
    run = f"{character}_s{seed}"
    try:
        e = _up.CHECKPOINTS["rl"][run][str(step)]
    except KeyError:
        raise KeyError(f"no released checkpoint for {run} step {step}; released: {released_steps(character, seed)}")
    return Checkpoint(character, seed, step, e["tinker"], None, e["hf"])


def released_steps(character: str, seed: int) -> list[int]:
    return sorted(int(s) for s in _up.CHECKPOINTS["rl"][f"{character}_s{seed}"])


def crossing_step(character: str, seed: int) -> int:
    """First released RL step (≈ where the training hack rate reached 50%).
    anti_s2 never learned to hack: its step 40 is not a real crossing."""
    return _up.crossing_step(f"{character}_s{seed}")


def crossing(character: str, seed: int) -> Checkpoint:
    return get(character, seed, crossing_step(character, seed))


def is_real_crossing(character: str, seed: int) -> bool:
    return (character, seed) != ("anti", 2)


def all_checkpoints() -> list[Checkpoint]:
    """All 30: 3 SFT + 9 runs × 3 released steps."""
    out = [sft(c) for c in CHARACTERS]
    for c in CHARACTERS:
        for s in SEEDS:
            out += [get(c, s, st) for st in released_steps(c, s)]
    return out


# Published held-out hack rates at the crossing (corrected grader), handoff §3. Used by M1 and
# as sanity references only; never as data.
PUBLISHED_CROSSING_HACK_RATE = {
    ("pro", 1): 0.545, ("pro", 2): 0.591, ("pro", 3): 0.473,
    ("neutral", 1): 0.555, ("neutral", 2): 0.491, ("neutral", 3): 0.455,
    ("anti", 1): 0.700, ("anti", 2): 0.000, ("anti", 3): 0.564,
}
