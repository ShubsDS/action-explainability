# costs/

`ledger.jsonl` is the append-only spending ledger that `td.cost` writes. It's committed to git,
and nobody edits it by hand. Each line is one event: `reserve`, `settle`, `release` or `reconcile`.
See the `td.cost` docstring.

At the end of each milestone:

1. Run `Budget("<M>").reconcile("tinker", billed_usd)` with the Tinker billing figure, and the
   same for OpenRouter.
2. Commit the ledger, and put the differences in the milestone report.

If a crash leaves reservations open, check provider billing by hand first. Then clear them with
`Budget("<M>").release_stale(rids, reason)`.
