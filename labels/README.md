# labels/

These are human labels, made by the user and never by Claude. They're committed to git.

- `<name>.batch.jsonl`: a blind batch made by `td.label_cli.make_batch`. It never changes once
  made. It shows only whitelisted fields, with no judge outputs, grades or detector flags.
- `<name>.labels.jsonl`: decisions, appended as you go. If you label an item twice, the last
  decision counts.

Label a batch, or check how far through it you are:

```bash
bin/td python -m td.label_cli labels/<name>.batch.jsonl
```

```bash
bin/td python -m td.label_cli labels/<name>.batch.jsonl --summary
```

Batches planned in the handoff:

| Batch | Milestone | Labels |
|---|---|---|
| 30 flagged accidental passes | M1 to M3 | accidental / intentional / unsure |
| 100 open + 100 silent candidates | M3 | open / silent / illegible / not_a_hack |
| 50 interrogation answers | M4 | admits / denies / unclear |

## Model reads

A `<name>.model_read.jsonl` file is **not a human label**. It is a classification made by Claude
at the user's request, and it must be cited as a model read. Each has a header line saying who
made it, by what method, and what it was checked against.

| File | What it is |
|---|---|
| `m1_silence.model_read.jsonl` | How the reasoning of each of the 80 M1 hacks treats the override (2026-11-14). The user declined to label `m1_silence_candidates.batch.jsonl`, which stays in place and is optional. |
| `m3_silence_flagged.model_read.jsonl` | What the reasoning does in each of the 21 attempts that a silence judge flagged (2026-11-20, second pass with context, code and grade). The user reviewed the grouping in chat and agreed with it; it was not a blind label. |

