"""Rendering to TOKENS via the upstream `nemotron3` renderer.

Single-turn prompts are built exactly as upstream builds them: `RewardHackEnv(...).initial_observation()`
with system prompt "You are Corin." and the test shown in the prompt (M2 acceptance test 1).

Forks are pure token arithmetic on a stored rollout (M2 acceptance test 2):
    prefix = parent.prompt_tokens + parent.output_tokens[:k] + insert_tokens
    max_tokens = MAX_TOKENS - len(prefix)
MAX_TOKENS (65,536) is in effect a TOTAL CONTEXT limit: M1 showed the server stops every rollout
at prompt + output = 65,536 even though upstream asks for max_tokens=65536. So a fork's room is
65,536 minus the whole prefix, prompt included — the handoff's literal formula. A forked
continuation therefore ends where its parent could have ended.

Multi-turn contexts (M4 C1/C2, M5b) are built by TOKEN CONCATENATION, never by re-rendering the
history: `Nemotron3Renderer` strips <think> from every assistant turn before the last user turn,
so the message API cannot express "reasoning kept" (NOTES.md, conflict 2).
"""
from __future__ import annotations

import functools

from td import config


@functools.lru_cache(maxsize=1)
def renderer():
    """Upstream's renderer for the base model (thinking on). Downloads the tokenizer once."""
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from src.tinker_local.tinker_sampling import get_renderer

    return get_renderer(config.RENDERER_NAME, get_tokenizer(config.BASE_MODEL))


def make_env(item: dict):
    """Upstream RewardHackEnv for an item, configured exactly like scripts/heldout_gen.py."""
    from src.train.rlaif.reward_hack.env import RewardHackEnv

    return RewardHackEnv(renderer=renderer(), sys_prompt=config.SYS_PROMPT, timeout=config.EXEC_TIMEOUT_S,
                         show_test_in_prompt=config.SHOW_TEST_IN_PROMPT, **item)


async def single_turn_prompt(item: dict) -> tuple[list[int], list, str]:
    """(prompt_tokens, stop_condition, user_prompt) for a first attempt at `item`."""
    env = make_env(item)
    obs, stop = await env.initial_observation()
    return list(obs.to_ints()), stop, env.user_prompt


def prompt_opens_thinking(prompt_tokens: list[int]) -> bool:
    from src.utils.parsing import prompt_opens_thinking as _p

    return _p(prompt_tokens, renderer())


def encode_insert(text: str) -> list[int]:
    """Tokenize insert text with no special tokens. NOTE: tokens at the seam between
    output_tokens[:k] and the insert may not equal tokenize(decode(prefix) + text); we keep the
    parent's tokens as they were and append the insert's own tokens (exact-fork semantics)."""
    return list(renderer().tokenizer.encode(text, add_special_tokens=False))


def decode(tokens: list[int]) -> str:
    return renderer().tokenizer.decode(tokens)


def fork_prefix(parent: dict, k: int, insert_tokens: list[int] | None = None) -> tuple[list[int], int]:
    """(model_input_tokens, max_tokens) for continuing `parent` after its first k output tokens."""
    out = parent["output_tokens"]
    if not 0 <= k <= len(out):
        raise ValueError(f"fork point k={k} outside [0, {len(out)}]")
    ins = list(insert_tokens or [])
    prefix = list(parent["prompt_tokens"]) + list(out[:k]) + ins
    budget = config.MAX_TOKENS - len(prefix)  # total-context limit (see module docstring)
    if budget <= 0:
        raise ValueError("fork leaves no output budget")
    return prefix, budget


def think_span(output_tokens: list[int]) -> tuple[int, int] | None:
    """Token index range [start, end) of the reasoning inside the output (before </think>),
    or None if </think> never appears. Used to keep M5 fork points inside <think>."""
    tok = renderer().tokenizer
    close = tok.encode("</think>", add_special_tokens=False)
    n = len(close)
    for i in range(len(output_tokens) - n + 1):
        if output_tokens[i:i + n] == close:
            return 0, i
    return None


def multi_turn_context(parent: dict, turns: list[dict], *, keep_reasoning: bool) -> list[int]:
    """Tokens for: parent's first turn (prompt + assistant output) → `turns` → generation prompt.

    keep_reasoning=True  (M4 C2): parent's raw output_tokens verbatim, <think> included.
    keep_reasoning=False (M4 C1): parent's output with the <think>…</think> span removed, in the
                                  exact form the renderer itself emits for historical turns.
    `turns`: later messages, e.g. [{"role": "user", "content": Q1}] or the M5b sequence.

    Built in M2; acceptance test 3 requires C1 != C2 and each to equal a hand-written expected string.
    """
    raise NotImplementedError("M2: multi-turn token rendering (see module docstring)")
