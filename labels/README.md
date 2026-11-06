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
