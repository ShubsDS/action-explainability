# prereg/

Freeze these before each milestone's main sampling (handoff rule 6):

- thresholds, prompts, insert texts and selection rules;
- sample sizes and analysis plans.

Freezing means:

1. Set `status: frozen` in `prereg/<milestone>.yaml`.
2. Commit it.
3. Tag it: `git tag prereg-<milestone>-v<N>`.

After that, any change gets a new version and a new tag, and a line in NOTES.md saying why.

Pilot data is labelled `pilot` in the rollout `condition` and never reused as main data.

| File | Milestone | Status |
|---|---|---|
| `m3.yaml` | M3 silence judges | draft (models not chosen: open question 4) |
| `m4.yaml` | M4 interrogation | draft (wording frozen after a pilot of ≤ 20 rollouts) |
| `m5.yaml` | M5 control, step 1: does the reasoning have a causal role? | draft (redesigned 2026-11-13; margin not set) |
| `phase2.yaml` | Phase 2 claim rule | not started (needs Gate 2) |
