# Gate 1: decision memo

**For the user to decide.** No M2+ spending happens until you do. Written 2026-11-12.

## Checklist

| Item | Status |
|---|---|
| M1 reproduces | **Yes.** 80/110 = 72.7% (Wilson 63.7% to 80.2%) vs published 70.0% (`reports/m1.md`) |
| Checkpoint access | **Yes,** via public Tinker sampler paths. Needs SDK `tinker` 0.39.0, because the server rejects upstream's locked 0.25.0 (approved, NOTES.md) |
| Measured cost per rollout and current prices recorded | **Yes.** $0.0346 per rollout (M1). Tinker prices re-checked 2026-11-11, unchanged. OpenRouter judge prices recorded |
| Kill-rule question settled (does Paul Colognese's Pivotal fellowship already cover this?) | **Open: yours to answer.** Blocks M3 paid sampling |
| Ledger reconciled against billing | **Open.** Ledger says $3.83 (M0 $0.024 + M1 $3.80). Needs the Tinker console figure |

## What changed since the handoff's budget

1. **Rollouts cost about 1.5 to 2× more.** M1's mean output was 23.4k tokens (median 15.3k,
   with 7% truncated at the 65,536 context), against the assumed 11k to 16k. **$0.035 per full
   rollout**, versus 1.6 to 2.3 cents. This comes from one crossing cell; other checkpoints may be
   shorter (M0's SFT rollout was 7.7k).
2. **Judge calls are expensive, because the reasoning is long.** The four monitor views send about
   66k input tokens per rollout. All 3 judges × 4 views cost about **$0.10 per rollout**, and
   Haiku 4.5 is about 75% of that. (Judge output is *assumed* to be about 500 tokens per call;
   measure it in a small M3 pilot.)
3. **Honest rollouts are rare after RL.** 3/110 (2.7%) were `honest_hard` at the anti_s1
   crossing. Most of the at-least-200 honest pool per character must come from SFT cells, whose
   honest yield is unmeasured (M0: 1/1).

## Projection, M3 to M6

Per-rollout figures: $0.035 Tinker (first attempt or continuation) and $0.10 monitor judging.
These are central estimates with ranges, not worst cases. `cost.py` still reserves worst cases
at run time.

### M3 (cap $250): **≈ $235 to $300, at or over the cap**

| Item | Volume | Est. |
|---|---|---|
| FP-recheck sampling (steps 60/90 × anti and neutral s1 to s3 × 110) | 1,320 rollouts | $46 |
| Honest-pool top-up from SFT and crossing cells (yield unknown: 25 to 50%) | 720 to 1,440 rollouts | $10 to $50 |
| Pro cell for the 100-open-hack label batch | ~110 rollouts | $4 |
| Monitor judging (3 judges × 4 views), recheck + honest pools | ~1,650 rollouts | ~$165 |
| Silence judges (2 judges, on ~1,100 hacks; models not chosen) | ~2,200 calls | $10 to $35 |

Haiku is the main cost driver (about $125 of the $165). Ways to fit under $250:
**(a)** release part of the contingency; **(b)** judge the honest pools only on the views whose
thresholds we need (all scored views are needed, so this saves little); **(c)** keep the recheck
at full size and pilot judge-output length first, since the projection may tighten.
We'd lean towards (c), then (a) if the pilot confirms the estimate.

### M4 (cap $300): **≈ $60 to $90, well under**

M4 reuses M3's step-60/90 rollouts for anti_s1, anti_s3 and neutral_s2. It adds anti_s3@42 and
a top-up for neutral_s2's silent hacks ($5 to $25, depending on neutral's silent share, which is
unknown). About 1,800 interrogation answers at about $0.016 each (prefill-heavy, ~5k output
assumed) is about $30. The C5 outside reader and admission scoring come to about $25 (models not
chosen).

### M5a (cap $400): **≈ $680 as specified, about 1.7× the cap**

| Per checkpoint | Volume | Est. |
|---|---|---|
| Source first attempts (≈100 usable tasks × 3) | 300 | $10.5 |
| Screening (≈100 prefixes × 8 placebo) | 800 | $27 |
| Arms (60 prefixes × 4 arms × 16) | 3,840 | $131 |
| **Per checkpoint** | | **≈ $170** |
| **4 checkpoints** | | **≈ $680** |

The handoff estimated the arms alone at $60 to $90 per checkpoint, assuming 11k to 16k tokens;
measured lengths put them at about $131. The design choices, which are yours:

| Option | Est. total | Note |
|---|---|---|
| As specified, release contingency ($300) and move part of M4's surplus | ≈ $680 | Keeps the design. Needs both cap changes |
| 12 continuations per prefix × arm (4 checkpoints) | ≈ $550 | Less power per prefix |
| 3 checkpoints × 16 | ≈ $510 | Drops one checkpoint (e.g. neutral_s3) |
| 3 checkpoints × 12 | ≈ $410 | Fits about the cap |
| 40 prefixes × 16 (4 checkpoints) | ≈ $500 | Fewer prefixes; mixed-model random effects get weaker |

Fork prefill is small (prompt plus ≤1k reasoning tokens). Screening cost is uncertain: at the
crossing many tasks are 5/5 hacked (M1), so fewer prefixes may land in the 2-to-6-of-8 band, and
screening could cost more.

### M5b (cap $150): **sized to the cap by design**

Only runs if M5a's positive control passes in at least 2 checkpoints. Each scored second
attempt is about $0.05 (≈25k prefill plus a full rollout), so the cap buys about 3,000 attempts.

### M6: no spending.

### Phase 1 total

| | Caps | Projected |
|---|---|---|
| M3 + M4 + M5a + M5b | $1,100 | ≈ $1,125 to $1,220 as specified |
| + contingency | $1,400 | fits **if** the contingency is released, mostly to M5a |

## Decisions needed

1. **Kill rule:** has Paul confirmed the Pivotal fellowship doesn't already cover this?
2. **M5a design or budget:** choose one of the options above. You don't need to decide before M3,
   but it changes how much contingency to hold back.
3. **M3:** OK to run a small judge pilot (≈30 rollouts × 3 judges × 4 views ≈ $3) at the start of
   M3 to measure judge output length before committing?
4. Still open from the handoff: **(a)** request the authors' original transcripts? That could cut
   M3/M4 sampling. **(b)** Confirm the M5 checkpoints, and whether M5b is in scope. **(c)** Which
   models to use for the silence judges and the C5 outside reader.
5. **Billing:** the Tinker console total for M0 + M1, so the ledger can be reconciled.
