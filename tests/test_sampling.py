"""Kill-and-resume never re-pays (handoff rule 7). Tinker and grading are mocked; the full M2
acceptance test 4 (real tokenizer + Docker, 2 tasks × 2 samples) is built in M2."""
import asyncio
from types import SimpleNamespace

import pytest

from td import config, cost, grading, render, sampling, store
from td.checkpoints import Checkpoint

CK = Checkpoint("anti", 1, 30, "tinker://fake/sampler_weights/000030", None, "hf/x")
ITEMS = [{"task_id": f"t{i}", "impossible_type": "conflicting", "test": "", "impossible": True}
         for i in range(2)]


@pytest.fixture
def mocked(tmp_data, monkeypatch):
    calls = {"sample": 0, "grade": 0}
    crash = {"on": False}

    async def fake_prompt(item):
        return [1, 2, 3], None, "user prompt"

    async def fake_sample(self, prompt_tokens, max_tokens):
        calls["sample"] += 1
        await asyncio.sleep(0.01)  # yield, so concurrent reservations are outstanding together
        return SimpleNamespace(tokens=[7] * 100, stop_reason="stop")

    async def fake_grade_and_store(self, spec, raw):
        calls["grade"] += 1
        if crash["on"]:
            raise KeyboardInterrupt("killed after paying, before grading")
        row = {f: None for f in store.ROLLOUT_FIELDS} | {"rollout_id": raw["rollout_id"],
                                                        "cost_usd": raw["cost_usd"]}
        self.rows.append(row)
        return row

    monkeypatch.setattr(grading, "require_docker", lambda *a, **k: None)
    monkeypatch.setattr(render, "single_turn_prompt", fake_prompt)
    monkeypatch.setattr(sampling.Sampler, "_sample", fake_sample)
    monkeypatch.setattr(sampling.Sampler, "_grade_and_store", fake_grade_and_store)
    return calls, crash


def _specs():
    return [sampling.SampleSpec(item=it, sample_idx=j) for it in ITEMS for j in range(2)]


def test_kill_and_resume_does_not_repay(mocked):
    calls, crash = mocked
    budget = cost.Budget("M2")

    crash["on"] = True  # first run: every sample is paid, then the process "dies" before grading
    s = sampling.Sampler(CK, budget, "dry", client=object())
    for spec in _specs():
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(s.run(spec))
    assert calls["sample"] == 4
    paid = budget.status()["settled"]
    assert paid == pytest.approx(4 * cost.tinker_cost(3, 100))
    assert budget.status()["outstanding"] == 0

    crash["on"] = False  # resume in a fresh process: grades from the raw journal, samples nothing
    s2 = sampling.Sampler(CK, budget, "dry", client=object())
    rows = asyncio.run(s2.run_all(_specs()))
    assert all(isinstance(r, dict) for r in rows)
    assert calls["sample"] == 4
    assert budget.status()["settled"] == pytest.approx(paid)

    s3 = sampling.Sampler(CK, budget, "dry", client=object())  # third run: fully cached
    asyncio.run(s3.run_all(_specs()))
    assert calls["sample"] == 4 and calls["grade"] == 8
    assert len(store.RolloutStore("dry")) == 4


def test_cap_stops_sampling(mocked, monkeypatch):
    calls, _ = mocked
    monkeypatch.setitem(config.CAPS, "M2", cost.tinker_rollout_worst_case(3) * 1.5)  # room for one
    s = sampling.Sampler(CK, cost.Budget("M2"), "capped", client=object())
    res = asyncio.run(s.run_all(_specs()))  # 4 concurrent: worst cases overlap, only one fits
    assert calls["sample"] == 1
    assert sum(isinstance(r, cost.CapExceeded) for r in res) == 3
