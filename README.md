# action-explainability: talker/doer experiments

This project asks whether what a model says controls what it does, using reward hacking on
Impossible-LiveCodeBench as the test case. The models are the Corin model organisms from
Colognese and Ward: Nemotron-3-Super-120B with anti, neutral or pro character SFT, followed by RL.

`NOTES.md` holds decisions, deviations, the upstream pin and measured prices. The research plan
and milestones (M0 to M8) are in the handoff document.

## Setup

1. Get the upstream code (submodule pinned in `NOTES.md`):

   ```bash
   git submodule update --init
   ```

2. Build upstream's frozen environment. We use it as-is:

   ```bash
   (cd upstream && uv sync --frozen)
   ```

3. Create `.env`, then add `TINKER_API_KEY` and `OPENROUTER_API_KEY` to it:

   ```bash
   cp .env.example .env
   ```

4. Start Docker. Grading refuses to run on any other sandbox backend.

## Running

`bin/td` runs a command inside upstream's environment, with `src/` on the path.

Run the tests (paid tests never run by default):

```bash
bin/td python -m pytest -q -rs
```

Print the plan and worst-case cost for a milestone without spending anything:

```bash
bin/td python -m td.experiments.m0_smoke --plan
```

Run it for real (spends money, within the M0 cap):

```bash
bin/td python -m td.experiments.m0_smoke
```

## Layout

```
upstream/        git submodule: barbonara/character-reward-hacking (grader, renderer, data, monitors)
src/td/          our code; thin wrappers around upstream, never copies of it
  config.py      paths, settings, prices, caps, kill-rule flag
  checkpoints.py tasks.py render.py sampling.py grading.py store.py cost.py
  judges/        monitors (upstream prompts), silence judges, admission scorer
  label_cli.py   blind hand-labelling tool
  experiments/   one script per milestone (all have --plan)
  analysis/      intervals, bootstrap, tests
prereg/          frozen, git-tagged preregistrations
labels/          human labels (committed)
reports/         milestone reports
costs/           append-only spending ledger (committed)
data/            paid rollouts and judgements (git-ignored; back up!)
```

## Ground rules enforced in code

- **Spending caps.** `td.cost.Budget` reserves each paid call's worst-case cost before
  dispatching it, and refuses anything that would pass the milestone cap.
- **Kill rule.** Reservations for M3 and later are refused until `td.config.KILL_RULE_SETTLED`
  is set.
- **Docker only.** `td.grading.require_docker()` refuses any sandbox backend other than Docker,
  whether it's set in the environment, in upstream's grader or in YAML.
- **No paying twice.** Paid samples are journaled the moment they arrive, so a crash never
  costs money twice (`td.sampling`, `td.store`).
