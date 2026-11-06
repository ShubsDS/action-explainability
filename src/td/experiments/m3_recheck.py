"""M3 (cap $250; BLOCKED until the kill-rule question is settled): monitors, silence judges, hand labels.

To build:
1. Honest pools per character (anti, neutral): `honest_hard` rollouts on CONFLICTING tasks from all
   sampled cells incl. step 0; target >= 200 each (top up with SFT + crossing rollouts; estimate first).
2. Thresholds per character x judge x view at 2% FPR (td.judges.monitors.per_character_thresholds);
   also realised FPR on honest rollouts from steps >= 30 only.
3. FP recheck: steps 60 and 90 of anti_s1..3 and neutral_s1..3, 110 impossible-side rollouts each;
   four views x three judges; catch rate, static detectors, silent share with task-clustered CIs.
4. Silence judges (prereg/m3.yaml) on all hacks.
5. Hand-label batch: 100 open hacks (mostly pro) + 100 silent candidates (anti/neutral), split 50/50
   by task (td.label_cli.split_by_task); STOP until the user labels.
"""
raise SystemExit("M3 is not built yet (and is blocked by the kill rule; see td.config.KILL_RULE_SETTLED)")
