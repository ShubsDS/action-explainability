# NOTES

Running log of decisions, deviations from the handoff, the pinned upstream SHA, and measured
costs/prices. Append; don't rewrite history. Date every entry.

## Upstream pin

| Item | Value |
|---|---|
| Repo | https://github.com/barbonara/character-reward-hacking |
| Submodule path | `upstream/` |
| Pinned SHA | `dd7148bb0773954720be0c26a0a45d429e5c7a11` (2026-10-12, "CHANGELOG: Fig 10 mean-rating interval…") |
| Pinned on | 2026-11-06 |

`td.config.UPSTREAM_SHA` records this value, and `tests/test_config.py` fails if the
checked-out submodule HEAD is different. To move the pin, update both, and write down why here.

## Environment (2026-11-06)

- Everything runs in **upstream's frozen environment** (`cd upstream && uv sync --frozen`), not one
  we resolve ourselves, so the grader, renderer and tokenizer versions match what the released
  models were evaluated with. Our extras (`requirements-extra.txt`, currently only statsmodels) are
  layered on top by `bin/td` with `uv run --with-requirements`.
  - `tests/test_env.py` checks that the overlay hasn't changed the version of any package upstream
    locks. The first overlay reinstalled numpy, but at the same version.
- Run things as `bin/td python -m ...`. `bin/td` puts `src/` on `PYTHONPATH`, and
  `td.config` puts `upstream/` on `sys.path`.
- Host: macOS arm64, Python 3.11.14 (uv-managed). The upstream lock supports darwin.
- `.env` lives at **our** repo root. `td.config` loads it before anything imports upstream.
  Upstream's `scripts/common.py` loads `upstream/.env` with `override=False`, so our values win.
  Don't create `upstream/.env`.
- Docker was **not running** when the scaffold was set up, so the "zero Docker skips" check (M0)
  hasn't been run yet.

## Conflicts between the handoff and the upstream code

The handoff says to trust the code when the two conflict. Each conflict is listed here.

1. **Training-state paths.** The handoff (§3) says every checkpoint is also on Tinker as a
   training state (`.../weights/...`). Upstream `configs/checkpoints.json` @ dd7148b lists only
   `tinker` (sampler, `.../sampler_weights/...`) and `hf` for each of the 30 checkpoints. So
   `Checkpoint.state_path` is `None` for now. Phase 1 needs no training, so nothing is blocked.
   Phase 2 starts from new SFT replicates, but if we ever need a released training state, ask the
   user or the authors.
2. **Thinking in history (`nemotron3` renderer).** `Nemotron3Renderer.render_message` strips
   thinking from every assistant message **before the last user message**, matching the HF
   template's `truncate_history_thinking`. So rendering `[user, assistant(with thinking), user(Q1)]`
   through the renderer always drops the first turn's `<think>`. That is the right result for
   M4 C1, but **C2 (reasoning kept) can't be produced through the message API**.
   `td.render` therefore builds multi-turn contexts by **token concatenation**: the parent's raw
   `prompt_tokens + output_tokens`, then the rendered later turns. It doesn't re-render the
   history. The M2 acceptance test 3 checks this.
3. **Monitor judges.** `scripts/monitor_judge.JUDGES` also has `sonnet: anthropic/claude-sonnet-5`
   alongside the three the handoff names. We use only haiku, qwen and deepseek, as the handoff says.
4. **Upstream's `select_rows`** drops error rows, infra errors, solvable rows, truncated rollouts
   (`stop_reason == "length"`) and empty-reasoning rows, in that order. `td.judges.monitors` applies
   the same filters to our store rows, so the counts can be compared.

5. **Broken training task.** The handoff says to scan the training tasks for the held-out defect
   (a bare `assert` with no `def check`). None have that defect, but **`lcbhard_77`'s `conflicting`
   test doesn't parse**: an unterminated string literal at line 27, so it looks truncated (36,443
   chars; its `original` parses). No program can pass it, so it's unusable too.
   `td.tasks.broken_reason` flags both kinds. Usable conflicting tasks: **20 held-out and 80
   training**. No `original` test is broken.

## Prices (to re-check in M0)

From the handoff, Tinker list prices on 2026-10-19, marked 50% off. **Not yet re-checked.**

| Item | $/M tokens |
|---|---|
| Sampling (output) | 1.44 |
| Prefill (input) | 0.57 |
| Cached prefill | 0.114 |
| Training | 1.276 |

These live in `td.config.PRICES`. M0 must confirm them against https://thinkingmachines.ai/tinker/pricing/,
update both places, and record the date and whether the discount still applies.

OpenRouter judge prices: TBD in M3, per model.

## Measured tokens per rollout

TBD (M0). The handoff budget assumed 11,000 to 16,000 output tokens per full rollout. The M0
measurement replaces this, and `cost.py` worst-case reservations always use the full 65,536 anyway.

## Data locations

- Paid rollouts and judgements: `data/` (git-ignored, append-only JSONL; see `td.store`).
  **Back this directory up**: it holds money already spent.
- Cost ledger: `costs/ledger.jsonl` (committed, append-only).
- Human labels: `labels/` (committed).

## Open questions for the user (handoff §8)

1. Has the kill-rule question been answered (rule 2)? It blocks M3 paid sampling.
2. Should we request the authors' original transcripts?
3. Confirm the M5 checkpoint set, and whether M5b is in scope.
4. Which OpenRouter models to use for the silence judges and the C5 outside reader.

## Log

- 2026-11-06: Scaffold created. Submodule pinned. Upstream env synced. No paid calls made.
