"""M4 (cap $300; needs prereg/m4.yaml frozen and git-tagged first): access and disclosure.

Cells: anti_s1@60/90, anti_s3@42/60, neutral_s2@60/90; ~100 silent hacks per seed. Honest controls from
the same cells plus anti_s2@60/90. Conditions C1–C6 (handoff §5, M4), one answer per rollout per
condition, same checkpoint, thinking on, T=1.0, 16,384-token answer budget. C1/C2 need
td.render.multi_turn_context (M2). Score with td.judges.admission; pilot on <= 20 rollouts, then freeze.
"""
raise SystemExit("M4 is not built yet")
