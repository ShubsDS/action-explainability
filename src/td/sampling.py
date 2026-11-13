"""Tinker sampling: first attempts, token-level forks, multi-turn contexts.

Same policy class and settings as upstream held-out eval (`TinkerTokenCompleter`, temperature 1.0,
65,536-token budget). Every call:
  1. computes the cache key; if the graded row exists, returns it (never re-pays);
  2. if a raw sample with that key is journaled (we paid, then crashed before grading), grades it;
  3. otherwise reserves the worst-case cost, samples, journals the raw sample (fsync), settles the
     actual cost, grades in Docker, and appends the graded row.

Two files per store name: `<name>.raw.jsonl` (paid samples, ungraded) and `<name>.jsonl` (graded rows).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from td import config, cost, grading, render, store
from td.checkpoints import Checkpoint

log = logging.getLogger(__name__)


class RawStore(store._JsonlStore):
    id_field = "rollout_id"

    def __init__(self, name: str, root=None):
        super().__init__((root or config.ROLLOUTS_DIR) / f"{name}.raw.jsonl")

    def append(self, row: dict) -> None:
        self._append(row)


@dataclass
class SampleSpec:
    """One sample to draw. `prompt_tokens` is the full model input (prompt, or fork prefix)."""
    item: dict                      # upstream task item (for grading)
    sample_idx: int
    condition: str = "base"
    parent_rollout_id: str | None = None
    fork_token: int | None = None
    insert_id: str | None = None
    prompt_tokens: list[int] | None = None   # None → single-turn first attempt, rendered here
    user_prompt: str | None = None           # the task message, when prompt_tokens is a custom prompt
    output_prefix: list[int] | None = None   # tokens already "in" the output (fork prefix + insert)
    max_tokens: int = config.MAX_TOKENS


class Sampler:
    def __init__(self, ckpt: Checkpoint, budget: cost.Budget, store_name: str, *,
                 concurrency: int = 16, client=None):
        grading.require_docker()  # fail before paying for anything
        self.ckpt = ckpt
        self.budget = budget
        self.rows = store.RolloutStore(store_name)
        self.raw = RawStore(store_name)
        self._sem = asyncio.Semaphore(concurrency)
        self._client = client  # injectable for tests (mock)

    # ------------------------------------------------------------ tinker plumbing
    def client(self):
        if self._client is None:
            import tinker

            self._client = tinker.ServiceClient().create_sampling_client(
                model_path=self.ckpt.sampler_path, base_model=config.BASE_MODEL)
        return self._client

    async def _sample(self, prompt_tokens: list[int], max_tokens: int):
        import tinker
        from tinker_cookbook.completers import TinkerTokenCompleter

        policy = TinkerTokenCompleter(self.client(), max_tokens=max_tokens, temperature=config.TEMPERATURE)
        return await policy(tinker.ModelInput.from_ints(prompt_tokens), render.renderer().get_stop_sequences())

    # ------------------------------------------------------------ public
    def key(self, spec: SampleSpec) -> dict:
        # A supplied input (fork prefix, multi-turn context) is hashed into the key; a first
        # attempt's prompt is fixed by the task, so its key stays as it always was.
        input_hash = None if spec.prompt_tokens is None else store.tokens_hash(spec.prompt_tokens)
        return store.cache_key(checkpoint=self.ckpt.key(), task_id=spec.item["task_id"],
                               split=spec.item["impossible_type"], condition=spec.condition,
                               parent_rollout_id=spec.parent_rollout_id, fork_token=spec.fork_token,
                               insert_id=spec.insert_id, sample_idx=spec.sample_idx,
                               sampler={"temperature": config.TEMPERATURE, "max_tokens": spec.max_tokens},
                               input_hash=input_hash)

    async def run(self, spec: SampleSpec) -> dict:
        key = self.key(spec)
        rid = store.rollout_id(key)
        if rid in self.rows:
            return self.rows.get(rid)
        async with self._sem:
            raw = self.raw.get(rid) or await self._pay_and_journal(spec, key, rid)
            return await self._grade_and_store(spec, raw)

    async def run_all(self, specs: list[SampleSpec]) -> list[dict | BaseException]:
        return await asyncio.gather(*(self.run(s) for s in specs), return_exceptions=True)

    # ------------------------------------------------------------ internals
    async def _pay_and_journal(self, spec: SampleSpec, key: dict, rid: str) -> dict:
        if spec.prompt_tokens is None:
            prompt_tokens, _, user_prompt = await render.single_turn_prompt(spec.item)
        else:
            prompt_tokens, user_prompt = list(spec.prompt_tokens), spec.user_prompt
        worst = cost.tinker_rollout_worst_case(len(prompt_tokens), spec.max_tokens)
        what = f"{self.ckpt.label} {spec.item['task_id']} {spec.condition} #{spec.sample_idx}"
        with self.budget.reserve(worst, what) as r:
            res = await self._sample(prompt_tokens, spec.max_tokens)
            usd = cost.tinker_cost(len(prompt_tokens), len(res.tokens))  # uncached: an upper bound
            raw = {"rollout_id": rid, "key": key, "prompt_tokens": prompt_tokens,
                   "output_prefix": list(spec.output_prefix or []), "sampled_tokens": list(res.tokens),
                   "stop_reason": res.stop_reason, "user_prompt": user_prompt, "cost_usd": usd,
                   "created_at": store.now_iso()}
            self.raw.append(raw)  # journal BEFORE settle: if settle fails we still hold the sample
            r.settle(usd, {"provider": "tinker", "n_prompt": len(prompt_tokens), "n_output": len(res.tokens)})
        return raw

    async def _grade_and_store(self, spec: SampleSpec, raw: dict) -> dict:
        from src.utils.parsing import action_is_truncated, parse_action_to_reasoning_and_response

        rend = render.renderer()
        # The scored output is everything after the original prompt: fork prefix + new tokens.
        output = raw["output_prefix"] + raw["sampled_tokens"]
        base_prompt = raw["prompt_tokens"][:len(raw["prompt_tokens"]) - len(raw["output_prefix"])]
        reasoning, response = parse_action_to_reasoning_and_response(output, rend)
        truncated = raw["stop_reason"] == "length" or action_is_truncated(
            output, rend, thinking_prefilled=render.prompt_opens_thinking(base_prompt))
        # A prefix that already closed the reasoning hides a reasoning block the model REOPENED
        # and never closed: upstream's check finds the prefix's `</think>` and passes it.
        reopened, unclosed = reopened_think(rend.tokenizer.decode(raw["sampled_tokens"])) \
            if raw["output_prefix"] else (False, False)
        truncated = truncated or unclosed
        g = await grading.grade(spec.item, response, truncated=truncated)
        if g.get("infra_error"):
            # Sandbox failure: nothing was graded. Keep only the raw journal entry so a resume
            # re-grades it; never lock an ungraded row into the append-only store.
            raise RuntimeError(f"sandbox infra error grading {raw['rollout_id']}; check `docker info`, then resume")
        code = grading.extract_code(response) if not truncated else ""
        acc = grading.accidental_pass(code, spec.item["test"], g, reasoning) if code else \
            {"accidental_pass": False, "reasons": ["no code"]}
        key = raw["key"]
        row = {
            "rollout_id": raw["rollout_id"], "checkpoint": key["checkpoint"], "task_id": spec.item["task_id"],
            "split": spec.item["impossible_type"], "heldout": _heldout(spec.item["task_id"]),
            "condition": spec.condition, "parent_rollout_id": spec.parent_rollout_id,
            "fork_token": spec.fork_token, "insert_id": spec.insert_id, "sample_idx": spec.sample_idx,
            "prompt_tokens": base_prompt, "output_tokens": output,
            "reasoning": reasoning, "answer_text": response, "code": code, "user_prompt": raw.get("user_prompt"),
            "truncated": truncated, "stop_reason": raw["stop_reason"], "reopened_think": reopened,
            "n_prompt_tokens": len(base_prompt), "n_output_tokens": len(output),
            "grade": g, "accidental_pass": acc["accidental_pass"], "accidental_pass_reasons": acc["reasons"],
            "cls": grading.upstream_cls(g, response),
            "sampler": key["sampler"], "upstream_sha": config.UPSTREAM_SHA,
            "created_at": raw["created_at"], "cost_usd": raw["cost_usd"],
        }
        self.rows.append(row)
        return row


def reopened_think(sampled_text: str) -> tuple[bool, bool]:
    """(reopened, unclosed) for the text sampled AFTER a prefix that closed the reasoning:
    reopened if it contains `<think>`; unclosed if its last `<think>` has no `</think>` after it,
    in which case there is no answer and the sample counts as truncated."""
    i = sampled_text.rfind("<think>")
    return i >= 0, i >= 0 and sampled_text.rfind("</think>") < i


def _heldout(task_id: str) -> bool:
    from td.tasks import is_heldout

    return is_heldout(task_id)
