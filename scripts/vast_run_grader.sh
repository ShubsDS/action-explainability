#!/usr/bin/env bash
# Runs LOCALLY: the main run of the scoring-fact experiment (td.experiments.m5_grader) against the
# queue on the rented machine. Every attempt is submitted at once, so the machine works through
# them whether or not this laptop stays awake; the stage is resumable (stored rows are free) and
# is retried while it reports errors. When every row is stored it stops the machine (disk kept).
# Needs `scripts/vast_local.sh tunnel` running. Logs: data/vast/main.<try>.log
set -u
cd "$(dirname "$0")/.."
LOG=data/vast
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%S) $*"; }
export TD_VLLM_RELAY_URL=http://127.0.0.1:8001 TD_VLLM_CONCURRENCY=2000
for i in $(seq 1 12); do
  say "main run, try $i"
  bin/td python -m td.experiments.m5_grader --stage arms > $LOG/main.$i.log 2>&1
  grep -e ' rows' -e 'ERROR' $LOG/main.$i.log | cut -c1-300 | tail -6
  if ! grep -q "errors; rerun\|Traceback\|CapExceeded\|NotFrozen\|SandboxRefused" $LOG/main.$i.log; then
    say "main run complete on try $i"
    scripts/vast_local.sh charged
    scripts/vast_local.sh stop
    bin/td python -m td.experiments.m5_grader --stage analyze > $LOG/main.analysis.txt 2>/dev/null
    say "analysis written to $LOG/main.analysis.txt"
    exit 0
  fi
  curl -s --max-time 10 http://127.0.0.1:8001/health > /dev/null || { say "queue unreachable (machine stopped or tunnel down); waiting 2 min"; sleep 120; }
done
say "NOT COMPLETE after 12 tries"
scripts/vast_local.sh charged
exit 1
