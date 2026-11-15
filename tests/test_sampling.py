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


def test_fork_key_follows_the_input_tokens(mocked):
    s = sampling.Sampler(CK, cost.Budget("M2"), "keys", client=object())
    assert "input_hash" not in s.key(sampling.SampleSpec(item=ITEMS[0], sample_idx=0))  # first attempt
    fork = dict(item=ITEMS[0], sample_idx=0, condition="K", parent_rollout_id="p", fork_token=2,
                insert_id="K1", max_tokens=100)
    a = s.key(sampling.SampleSpec(**fork, prompt_tokens=[1, 2, 3, 9]))
    b = s.key(sampling.SampleSpec(**fork, prompt_tokens=[1, 2, 3, 8]))  # insert text edited, same insert_id
    assert store.rollout_id(a) != store.rollout_id(b)
    assert store.rollout_id(a) == store.rollout_id(s.key(sampling.SampleSpec(**fork, prompt_tokens=[1, 2, 3, 9])))


def test_an_analysis_keeps_only_rows_of_the_current_specs(mocked):
    from td.experiments import m5_reasoning as m5

    s = sampling.Sampler(CK, cost.Budget("M2"), "ids", client=object())
    old = sampling.SampleSpec(item=ITEMS[0], sample_idx=0, condition="ins_reject", parent_rollout_id="p",
                              fork_token=2, insert_id="reject", prompt_tokens=[1, 2, 3, 9], max_tokens=100)
    new = sampling.SampleSpec(item=ITEMS[0], sample_idx=0, condition="ins_reject", parent_rollout_id="p",
                              fork_token=2, insert_id="reject:0", prompt_tokens=[1, 2, 3, 8], max_tokens=100)
    assert sampling.spec_id(CK, new) == store.rollout_id(s.key(new)) != sampling.spec_id(CK, old)
    rows = [{"rollout_id": sampling.spec_id(CK, sp), "condition": sp.condition} for sp in (old, new)]
    keep, left_out = m5.current_rows(rows, [new], CK)   # same arm name, earlier wording: left out
    assert [r["rollout_id"] for r in keep] == [sampling.spec_id(CK, new)] and left_out == 1


def test_a_poisoned_client_is_dropped_so_later_samples_get_a_fresh_one(mocked):
    s = sampling.Sampler(CK, cost.Budget("M2"), "poison", client=object())
    s._drop_client_if_poisoned(RuntimeError("rate limited"))
    assert s._client is not None
    s._drop_client_if_poisoned(RuntimeError("400 - This SamplingClient is poisoned. completed results expired"))
    assert s._client is None


def test_cap_stops_sampling(mocked, monkeypatch):
    calls, _ = mocked
    monkeypatch.setitem(config.CAPS, "M2", cost.tinker_rollout_worst_case(3) * 1.5)  # room for one
    s = sampling.Sampler(CK, cost.Budget("M2"), "capped", client=object())
    res = asyncio.run(s.run_all(_specs()))  # 4 concurrent: worst cases overlap, only one fits
    assert calls["sample"] == 1
    assert sum(isinstance(r, cost.CapExceeded) for r in res) == 3
