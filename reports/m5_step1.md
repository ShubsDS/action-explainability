# M5 step 1: does a statement in the reasoning change whether the answer hacks?

**Status: run complete (2026-11-15).** Preregistration: `prereg/m5.yaml`, section
`reasoning_role`, frozen at tag `prereg-m5-v1` before any main sampling. Reproduce the tables with
`bin/td python -m td.experiments.m5_reasoning --stage analyze`.

## Result

| Checkpoint | Primary contrast | Difference in attempt rate | Preregistered reading |
|---|---|---|---|
| anti, step 0 (before RL) | "I will add the override" − placebo | **+66.9 pp** (90% interval +59.5 to +74.4; 19 tasks) | Before RL, a stated plan is carried out. |
| anti_s1, step 30 (hacks) | "I will not touch `check`" − placebo | **−1.9 pp** (90% interval −4.3 to +0.6; 99 tasks) | The answer hacks whatever the reasoning just committed to. |

The margin was 15 pp. The step-30 interval also lies inside ±5 pp.

## What was run

Every arm fixes the reasoning, closes it, and samples only the answer. An insert is one paragraph
added at the end of the reasoning, in the model's own words, in two wordings used on alternate
samples. The outcome is whether the answer attempts the hack (it rebinds the test's `check`, or
hacks some other way), among samples that gave an answer with code.

| | Hacking checkpoint (anti_s1@30) | Before RL (anti, step 0) |
|---|---|---|
| Tasks | 99: 20 held-out + 79 training | 19 held-out |
| Parents (first attempts whose reasoning is reused) | 316: 90 from M1 + 226 new on training tasks | 79 honest step-0 first attempts |
| Answers per parent per arm | 4 (held-out), 2 (training) | 4 |
| Answer samples | 2,752 | 948 |

First attempts sampled for this run: 240 on the 80 usable training tasks at step 30 (14 truncated;
217 of the 226 completed ones attempted the hack), and 100 on the 20 usable held-out tasks at step 0
(17 truncated; 4 of the 83 completed ones attempted it). One training task (lcbhard_94) had no
completed parent and one held-out task (lcbhard_71) had no usable step-0 donor, so they drop out.

## All arms

Attempt rate among answered samples; interval = the wider of a task-cluster bootstrap and Wilson.

| Arm | anti_s1@30 | anti, step 0 |
|---|---|---|
| Own reasoning, answer resampled | 771/808 = 95.4% (93.7 to 97.2) | 5/316 = 1.6% (0.0 to 3.6) |
| + placebo paragraph | 745/810 = 92.0% (89.4 to 94.4) | 2/315 = 0.6% (0.0 to 2.3) |
| + "I will not touch `check`" | 724/808 = 89.6% (86.4 to 92.5) | not run (floor) |
| + "I will add the override" | not run (ceiling) | 224/316 = 70.9% (63.7 to 77.1) |
| Honest step-0 reasoning swapped in | 297/316 = 94.0% (89.9 to 96.9) | n/a |

Lost samples were negligible: at most 5 of about 810 per arm (truncated or without code), and one
sample in the whole run reopened its reasoning.

## Contrasts

Mean within-task difference, every task weighted equally; 90% two-sided interval.

| Contrast | Difference | Interval | Tasks |
|---|---|---|---|
| **reject − placebo, step 30 (primary)** | −1.9 pp | −4.3 to +0.6 | 99 |
| held-out tasks only | −3.4 pp | −7.2 to +0.3 | 20 |
| training tasks only | −1.5 pp | −4.6 to +1.6 | 79 |
| working hacks only (secondary outcome) | −0.6 pp | −3.9 to +2.8 | 99 |
| **plan − placebo, step 0 (primary)** | +66.9 pp | +59.5 to +74.4 | 19 |
| working hacks only (secondary outcome) | +54.0 pp | +45.2 to +62.7 | 19 |
| placebo − own, step 30 (check) | −4.6 pp | −6.7 to −2.4 | 99 |
| placebo − own, step 0 (check) | −0.5 pp | −3.6 to +2.6 | 19 |
| donor − own, step 30 | −3.1 pp | −10.7 to +4.5 | 19 |

- **Check passed.** The placebo differs from no insert by less than the margin at both checkpoints,
  so the insert arms are interpretable. At step 30 any inserted paragraph ending "just the
  function" lowers the attempt rate by about 5 pp; the explicit commitment adds about 2 pp to that.
- **Donor swap.** After an honest step-0 reasoning the hacking checkpoint still attempts the hack
  94% of the time. Its own reasoning is not necessary for the hack. (Preregistered wording: this is
  not a claim that reasoning has no causal role.)

## Preplanned splits (exploratory)

| Split | Own | Placebo | Reject |
|---|---|---|---|
| Parent attempted on its first try (301 parents) | 97.8% | 93.9% | 91.8% |
| Parent did not (15 parents) | 21/41 = 51.2% | 23/41 = 56.1% | 21/42 = 50.0% |
| Parent's reasoning mentions defining `check` (144 parents; regex) | 93.6% | 89.6% | 87.7% |
| It does not (172 parents) | 97.0% | 94.0% | 91.2% |

| Wording | Reject, step 30 | Plan, step 0 |
|---|---|---|
| Wording 1 (longer) | 357/403 = 88.6% | 98/158 = 62.0% |
| Wording 2 (shorter) | 367/405 = 90.6% | 126/158 = 79.7% |

## Surprises

1. **The size of the gap.** The same kind of sentence moves the action by 67 points before RL and
   by 2 points after it.
2. **Reasoning that led to no hack still yields one half the time.** For the 15 parents that did
   not attempt on their first try, a resampled answer after the same reasoning attempts the hack
   51% of the time. Which answer follows a given reasoning is close to a coin flip there.
3. **The commitment is not what stops the minority.** Among non-attempting parents the reject
   insert does no better than the placebo (50% vs 56%).
4. **Per-task effects are small and scattered.** 47 of 99 tasks show exactly no difference between
   reject and placebo; 8 show a drop of 25 pp or more, on 6 to 20 answers each.

## Limits

- **The two primary contrasts are not the same manipulation.** Step 0 tests "says it will, then
  does"; step 30 tests "says it will not, then does anyway". The mirror images cannot be run: the
  hack is at ceiling at step 30 and at floor at step 0. So the result is also consistent with
  "a sentence can trigger an action but cannot suppress a trained habit", at any stage of training.
- **The plan insert names the exact line of code**; the reject insert asks for an omission.
- **Inserts are not the model's own continuation**, even in its own words (Macar et al., arXiv
  2510.27484). The step-0 result shows an insert can move the action, which bounds this concern.
- **One character, one seed, one step.** The step-0 side is 19 held-out tasks; the step-30 side
  leans on 79 tasks the checkpoint was trained on, though the two strata agree.
- **No human labels** anywhere in this result; the outcome is a static check on the code.

## Cost

| Item | Settled |
|---|---|
| 240 training-task first attempts (step 30) | $6.01 |
| 100 step-0 first attempts | $4.41 |
| 2,752 answer samples at step 30 | $30.38 |
| 948 answer samples at step 0 | $13.68 |
| **Step 1 main run** | **$54.48** (estimate was about $64) |

M5a total so far, pilots included: $58.62 of the $400 cap. 16 reservations ($0.38 worst case) from
the re-pilot's lost samples are still open; the ledger is not reconciled against Tinker billing.
