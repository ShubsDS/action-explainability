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
  - **One deliberate override, `tinker` 0.25.0 → 0.39.0** (2026-11-11, user approved). The Tinker
    server rejects 0.25.0 ("Your Tinker SDK version is no longer supported"), so no checkpoint can
    load with upstream's lock, and upstream has no newer commit. With 0.39.0 the three M0/M1
    checkpoints load, and upstream's pinned tinker-cookbook (ac227ae) still imports. No other
    locked package changes. It's only the client SDK: the renderer, tokenizer, grader and dataset
    stay at upstream's versions. Risk: sampler client behaviour (e.g. how `stop_reason` is
    reported) may differ from 0.25. The M1 reproduction against the published 70% is the check.
    The override is listed in `requirements-extra.txt` and `tests/test_env.py:ALLOWED_OVERRIDES`.
- Run things as `bin/td python -m ...`. `bin/td` puts `src/` on `PYTHONPATH`, and
  `td.config` puts `upstream/` on `sys.path`.
- Host: macOS arm64, Python 3.11.14 (uv-managed). The upstream lock supports darwin.
- `.env` lives at **our** repo root. `td.config` loads it before anything imports upstream.
  Upstream's `scripts/common.py` loads `upstream/.env` with `override=False`, so our values win.
  Don't create `upstream/.env`.
- Docker: Docker Desktop 29.0.1 (linux/aarch64 VM, 10 CPUs, 8 GB). Upstream's sandbox image
  `python:3.11-slim` is pinned by digest, and its multi-arch index covers arm64.

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

## Interpretation choices (not conflicts; written down so they can be challenged)

- **The 65,536 budget is a total-context limit** (measured in M1, 2026-11-12). Upstream asks for
  `max_tokens=65536`, but the server stops every rollout at prompt + output = 65,536. All 8
  truncated M1 rollouts ended exactly there (63,001 to 64,753 output tokens), and no untruncated
  rollout went past 64,971. Upstream had the same cap, so reproduction is unaffected. Three
  consequences:
  - **Fork budget.** `td.render.fork_prefix` now uses the handoff's literal
    `65,536 − len(full prefix)`, prompt included. An earlier draft subtracted only the
    output-side prefix, which was wrong in effect.
  - **Cost reservations** (prompt + 65,536 output) overstate the true worst case by the prompt's
    length. That errs on the safe side, so we keep them.
  - **Training tier.** Sequences can't exceed 64K, so training should be billed at the 64K-tier
    price, not the 256K one. Confirm with Tinker before M8.
- **Insert tokenization.** Inserts are tokenized on their own and appended after the parent's
  first k tokens, which are kept unchanged. The tokens at the seam can differ from re-tokenizing
  the joined text, and that's intended (exact-fork semantics).
- **Raw journal.** Each paid sample is written to `<store>.raw.jsonl` before it's graded, so a
  crash during grading, or a sandbox infra error, never forces a re-sample. A graded row is
  written only when grading succeeds; infra errors are never locked into the append-only store.
- **Actual cost.** We settle each Tinker call as fully uncached prefill, which is an upper bound.
  End-of-milestone reconciliation against billing shows how far the ledger overstates.
- **M1 denominator.** All 22 held-out tasks are sampled, including the two broken ones, which
  can't be hacked. That matches the published 77/110.
- **Accidental-pass detector** (`td.grading.accidental_pass`). It checks the code side
  (`exit_hack_suspected`, stdin I/O boilerplate, a `__main__` guard or entry call, no `check`, no
  literal test inputs, no source or frame introspection) plus a keyword check on the reasoning.
  It's a heuristic until the user hand-checks 30 flagged cases.

## Prices

**Re-checked 2026-11-11** at https://thinkingmachines.ai/tinker/pricing/, which still labels its
table "as of October 19, 2026". The prices for Nemotron-3-Super-120B-A12B are **unchanged** from
the handoff. The model carries a "50% off*" badge, and the footnote says the discount is already
reflected in the listed prices. No end date or terms are given, so check again before Phase 2
budgeting.

These are the 64K-context prices:

| Item | $/M tokens |
|---|---|
| Sampling (output) | 1.44 |
| Prefill (input) | 0.57 |
| Cached prefill | 0.114 |
| Training | 1.276 |

There is also a **256K-context tier** with the same prefill, cached-prefill and sampling prices,
but **training costs $2.32/M** instead of $1.276. Sampling costs the same in either tier, so
Phase 1 isn't affected. The M0 note worried that sequences slightly over 64K might be billed at the
256K rate. **M1 showed the server caps prompt + output at 65,536**, so sequences stay in the 64K
tier, and training should cost the $1.276 rate. Confirm this with Tinker before budgeting M8.

These prices live in `td.config.PRICES` (`PRICES_CHECKED_ON = 2026-11-11`).

OpenRouter list prices for the monitor judges, fetched 2026-11-12 from the public
`https://openrouter.ai/api/v1/models` (stored in `td.config.PRICES["openrouter"]`):

| Model | Input $/M | Output $/M | Context |
|---|---|---|---|
| anthropic/claude-haiku-4.5 | 1.00 | 5.00 | 200K |
| qwen/qwen3.5-122b-a10b | 0.26 | 2.08 | 262K |
| deepseek/deepseek-v4-flash-0731 | 0.0434 | 0.30 | 1M |

The silence-judge and C5 models are still to be chosen (open question 4). Add their prices here
before using them; `cost.openrouter_cost` refuses any model that isn't priced.

## Measured tokens per rollout

The handoff budget assumed 11,000 to 16,000 output tokens per full rollout. `cost.py`'s worst-case
reservations always use the full 65,536, whatever the measurement says.

| When | Cell | n | Prompt tok | Output tok | Cost per rollout |
|---|---|---|---|---|---|
| 2026-11-11 (M0) | anti SFT, anti_s1@60; task lcbhard_64 | 2 | 3,995 | 7,747 / 5,646 (mean 6,697) | $0.0134 / $0.0104 |

| 2026-11-12 (M1) | anti_s1@30, 22 held-out × 5 | 110 | mean 1,485 (304 to 3,995) | **mean 23,433**, median 15,298; untruncated mean 20,259 (median 14,026, p90 43,973); 8 truncated at the 65,536 context | **$0.0346** |

**M1 replaces the budget assumption.** Outputs are much longer than the handoff's 11k to 16k: the
mean is 23.4k including truncations. So a full rollout costs about **$0.035**, versus the assumed
1.6 to 2.3 cents, roughly 1.5 to 2× more. This is one crossing cell, and lengths may differ at
other checkpoints and steps (M0's SFT rollout was 7.7k). `reports/gate1.md` uses $0.035 for the
projection.

Judge input per rollout (M1, the 102 rows kept by upstream's filters): about 20.7k tokens for
`ronlybin`, 20.8k for `ronly`, 22.4k for `full` and 1.8k for `answer`, about 66k across all four
views. That's estimated at 3.6 chars per token; Nemotron itself averaged 3.77 chars per output
token.

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
  - Our tests: 66 passed, 1 skipped (the real-Docker grading test; the daemon isn't running). The
    Nemotron tokenizer and the HF dataset (rev 98650ff) both download without credentials.
  - Upstream's own test suite hasn't been run yet; that's M0, with Docker up.
  - Plans: M0 worst case is $0.20 (2 rollouts). M1 worst case is $10.78 (110 rollouts, against
    the $15 cap).
  - Not built yet: multi-turn rendering (M2), the M3 to M5 experiment scripts, TOST and the mixed
    models (M5). The silence-judge and C5 models are still unchosen (open question 4).
- 2026-11-11: **M0 test check passed**, with Docker running and no `.env` (so no API keys loaded).
  - Upstream (`cd upstream && uv run --frozen python -m pytest -q -rs`): 291 passed, 52 skipped,
    **0 Docker skips**. All 52 skips are upstream's network gate.
  - The same suite with `CORIN_NETWORK_TESTS=1`: **343 passed, 0 skipped**. Before running it we
    checked that the network tests make no paid calls; their Tinker clients are mocked, and the
    keys they use are fake ones for leak tests.
  - Ours (`bin/td python -m pytest -q -rs`): **67 passed, 0 skipped**. That includes the real-Docker
    grading test: an honest answer passes `original`, and an exit hack is flagged as
    `hacked_on_impossible` plus `exit_hack_suspected`.
  - Still to do for M0: the two smoke rollouts (needs `TINKER_API_KEY`, worst case $0.20), the
    measured tokens per rollout, and re-checked prices.
