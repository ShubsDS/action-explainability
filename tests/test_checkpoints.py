import pytest

from td import checkpoints as ck

# handoff §3 table
CROSSING = {("pro", 1): 20, ("pro", 2): 24, ("pro", 3): 21,
            ("neutral", 1): 38, ("neutral", 2): 42, ("neutral", 3): 25,
            ("anti", 1): 30, ("anti", 2): 40, ("anti", 3): 42}


def test_thirty_checkpoints():
    a = ck.all_checkpoints()
    assert len(a) == 30
    assert len({c.sampler_path for c in a}) == 30
    assert all(c.sampler_path.startswith("tinker://") and "/sampler_weights/" in c.sampler_path for c in a)


@pytest.mark.parametrize("char,seed", list(CROSSING))
def test_released_steps(char, seed):
    assert ck.released_steps(char, seed) == [CROSSING[(char, seed)], 60, 90]
    assert ck.crossing_step(char, seed) == CROSSING[(char, seed)]


def test_sft_is_step_zero_shared_across_seeds():
    assert ck.get("anti", 1, 0) == ck.get("anti", 3, 0) == ck.sft("anti")
    assert ck.sft("anti").seed is None


def test_unknown_step_and_character():
    with pytest.raises(KeyError):
        ck.get("anti", 1, 31)
    with pytest.raises(ValueError):
        ck.get("genuine", 1, 30)


def test_anti_s2_not_a_real_crossing():
    assert not ck.is_real_crossing("anti", 2)
    assert ck.is_real_crossing("anti", 1)


def test_a_rebuilt_copy_changes_where_we_sample_not_which_checkpoint_it_is(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(ck, "REBUILT_PATH", tmp_path / "rebuilt.json")  # no file: nothing rebuilt
    orig = ck.get("anti", 1, 30)
    assert orig.served_path is None and orig.model_path == orig.sampler_path

    entry = {"published_path": orig.sampler_path, "hf_repo": orig.hf_repo, "served_path": "tinker://ours/sampler_weights/x"}
    ck.REBUILT_PATH.write_text(json.dumps({"anti_s1@30": entry}))
    c = ck.get("anti", 1, 30)
    assert c.model_path == "tinker://ours/sampler_weights/x"
    assert c.key() == orig.key() and c.sampler_path == orig.sampler_path   # cache keys and records unchanged
    assert ck.get("anti", 2, 60).served_path is None                       # other checkpoints untouched

    ck.REBUILT_PATH.write_text(json.dumps({"anti_s1@30": entry | {"published_path": "tinker://other/sampler_weights/000030"}}))
    with pytest.raises(RuntimeError):
        ck.get("anti", 1, 30)
