"""M5 step 2 (conversation ladder): which prompts each arm builds, and how stored rows are read.
The renderer is faked; tests/test_render.py checks the real one. Nothing here spends money."""
import pytest

from td import render, store
from td.experiments import m5_ladder as ladder

ITEMS = {f"t{i}": {"task_id": f"t{i}", "impossible_type": "conflicting"} for i in range(6)}


@pytest.fixture
def fake_prompt(monkeypatch):
    calls = []

    def conversation_prompt(item, *, before=(), user_suffix=""):
        calls.append({"task": item["task_id"], "before": list(before), "suffix": user_suffix})
        return [1, 2, len(calls)], f"task {item['task_id']}{user_suffix}"

    monkeypatch.setattr(render, "conversation_prompt", conversation_prompt)
    return calls


def test_each_arm_changes_one_thing_and_has_a_placebo_of_the_same_shape(fake_prompt):
    cfg = ladder.load_cfg()
    specs = ladder.arm_specs(cfg, ITEMS, pilot=False)
    n = cfg["samples_per_task"]
    assert len(specs) == len(ITEMS) * len(cfg["arms"]) * n
    by = {(c["task"], i % len(cfg["arms"])): c for i, c in enumerate(fake_prompt)}
    t0 = dict(zip(cfg["arms"], (by[("t0", i)] for i in range(len(cfg["arms"])))))
    # earlier-turn arms: a user question and the model's answer, and nothing added to the task
    for arm in ("placebo_turn", "own_promise"):
        assert [m["role"] for m in t0[arm]["before"]] == ["user", "assistant"] and t0[arm]["suffix"] == ""
    assert "`check`" in t0["own_promise"]["before"][1]["content"]
    assert "check" not in t0["placebo_turn"]["before"][0]["content"] + t0["placebo_turn"]["before"][1]["content"]
    # instruction arms: one sentence appended to the task message, and no earlier turn
    for arm in ("user_placebo", "user_prohibition"):
        assert t0[arm]["before"] == [] and t0[arm]["suffix"].startswith("\n\n")
    assert "`check`" in t0["user_prohibition"]["suffix"] and "check" not in t0["user_placebo"]["suffix"]
    # every spec is a full first attempt with its task message recorded
    assert all(s.max_tokens == specs[0].max_tokens and s.output_prefix is None and s.user_prompt for s in specs)
    assert {s.condition for s in specs} == set(cfg["arms"])


def test_pilot_is_small_and_stored_apart(fake_prompt):
    cfg = ladder.load_cfg()
    specs = ladder.arm_specs(cfg, ITEMS, pilot=True)
    assert len(specs) == cfg["pilot"]["tasks"] * len(cfg["arms"]) * cfg["pilot"]["samples"]
    assert all(s.condition.startswith("pilot:") for s in specs)
    plain = ladder.plain_specs(cfg, ITEMS, pilot=True)
    assert {s.condition for s in plain} == {"pilot"} and all(s.prompt_tokens is None for s in plain)
    assert {s.condition for s in ladder.plain_specs(cfg, ITEMS, pilot=False)} == {"base"}


def test_stored_rows_are_split_into_pilot_and_main(tmp_data):
    s = store.RolloutStore("x")
    for rid, cond in (("a", "own_promise"), ("b", "pilot:own_promise"), ("c", "base"), ("d", "pilot")):
        s.append({f: None for f in store.ROLLOUT_FIELDS} | {"rollout_id": rid, "condition": cond})
    assert sorted(r["condition"] for r in ladder._rows("x", pilot=False)) == ["base", "own_promise"]
    assert sorted(r["condition"] for r in ladder._rows("x", pilot=True)) == ["base", "own_promise"]
    assert {r["rollout_id"] for r in ladder._rows("x", pilot=True)} == {"b", "d"}
    assert [r["condition"] for r in ladder._rows("x", pilot=False, rename={"base": "floor"}) if r["rollout_id"] == "c"] == ["floor"]


def test_prereg_contrasts_name_real_arms():
    cfg = ladder.load_cfg()
    arms = set(cfg["arms"]) | {"baseline"}
    for arm, base in cfg["contrasts"].items():
        assert {arm, base} <= arms
    for pair in cfg["primary"]:
        assert cfg["contrasts"][pair[0]] == pair[1]
    for a in cfg["arms"].values():
        assert set(a) <= {"before", "user_suffix"} and (a.get("before") or a.get("user_suffix"))
