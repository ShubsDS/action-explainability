import json

import pytest

from td import config, cost


def test_tinker_cost_arithmetic():
    p = {"sample_output": 1.44, "prefill": 0.57, "prefill_cached": 0.114, "train": 0}
    assert cost.tinker_cost(1_000_000, 1_000_000, prices=p) == pytest.approx(1.44 + 0.57)
    assert cost.tinker_cost(1_000_000, 0, n_cached_prompt=1_000_000, prices=p) == pytest.approx(0.114)


def test_worst_case_uses_full_budget():
    assert cost.tinker_rollout_worst_case(0) == pytest.approx(config.MAX_TOKENS * 1.44e-6)


def test_reserve_settle_release(tmp_data):
    b = cost.Budget("M0")
    r = b.reserve(1.0, "x")
    assert b.status()["outstanding"] == pytest.approx(1.0)
    r.settle(0.25)
    s = b.status()
    assert s["settled"] == pytest.approx(0.25) and s["outstanding"] == 0
    with b.reserve(2.0, "cache hit"):
        pass  # clean exit without settle → released
    assert b.status()["outstanding"] == 0


def test_cap_refuses(tmp_data):
    b = cost.Budget("M0")  # $10
    b.reserve(9.0, "a")
    with pytest.raises(cost.CapExceeded):
        b.reserve(1.5, "b")


def test_crash_keeps_reservation_open(tmp_data):
    b = cost.Budget("M0")
    with pytest.raises(ZeroDivisionError):
        with b.reserve(3.0, "crashy"):
            1 / 0
    s = b.status()
    assert s["outstanding"] == pytest.approx(3.0) and s["n_open"] == 1
    b.release_stale(s["open_rids"], "checked billing by hand")
    assert b.status()["outstanding"] == 0


def test_settle_above_worst_case_is_an_error(tmp_data):
    r = cost.Budget("M0").reserve(0.1, "x")
    with pytest.raises(RuntimeError):
        r.settle(0.2)


def test_kill_rule_blocks_m3(tmp_data):
    with pytest.raises(cost.KillRuleBlocked):
        cost.Budget("M3", kill_rule_settled=False).reserve(0.01, "x")
    cost.Budget("M3", kill_rule_settled=True).reserve(0.01, "x")  # allowed once settled


def test_ledger_is_append_only_jsonl(tmp_data):
    b = cost.Budget("M1")
    b.reserve(0.5, "x").settle(0.1)
    lines = config.LEDGER_PATH.read_text().splitlines()
    assert [json.loads(x)["type"] for x in lines] == ["reserve", "settle"]


def test_milestones_independent(tmp_data):
    cost.Budget("M0").reserve(9.9, "x")
    cost.Budget("M1").reserve(14.9, "y")  # M0 spend doesn't count against M1


def test_unpriced_judge_model_refused():
    with pytest.raises(KeyError):
        cost.openrouter_cost("nobody/unpriced-model", 10, 10)
