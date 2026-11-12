import pytest

from td import store

CK = {"character": "anti", "seed": 1, "step": 30, "sampler_path": "tinker://x"}
SM = {"temperature": 1.0, "max_tokens": 65536}


def _key(**kw):
    base = dict(checkpoint=CK, task_id="t1", split="conflicting", sample_idx=0, sampler=SM)
    return store.cache_key(**{**base, **kw})


def test_cache_key_stable_and_distinct():
    assert store.rollout_id(_key()) == store.rollout_id(_key())
    ids = {store.rollout_id(_key(**kw)) for kw in
           [{}, {"sample_idx": 1}, {"condition": "K"}, {"fork_token": 10}, {"insert_id": "P1"},
            {"parent_rollout_id": "abc"}, {"split": "original"},
            {"sampler": {"temperature": 1.0, "max_tokens": 1000}}]}
    assert len(ids) == 8


def test_input_hash_separates_inputs_but_leaves_first_attempt_ids_alone():
    # A first attempt (no input_hash) keeps the key, and so the id, it had before the field existed.
    # The pinned id was computed with the code as it stood when M0 and M1 were sampled.
    assert "input_hash" not in _key()
    assert store.rollout_id(_key()) == "7a58dff1a4b2d8e5e83e30487a14ce8d"
    fork = dict(parent_rollout_id="abc", fork_token=10, insert_id="K1")
    a = _key(**fork, input_hash=store.tokens_hash([1, 2, 3]))
    b = _key(**fork, input_hash=store.tokens_hash([1, 2, 4]))  # same insert_id, different text
    assert len({store.rollout_id(k) for k in (_key(**fork), a, b)}) == 3
    assert store.tokens_hash([1, 2, 3]) == store.tokens_hash((1, 2, 3))


def test_cache_key_ignores_extra_checkpoint_fields():
    assert store.rollout_id(_key()) == store.rollout_id(_key(checkpoint={**CK, "hf_repo": "x"}))


def _row(rid):
    return {f: None for f in store.ROLLOUT_FIELDS} | {"rollout_id": rid}


def test_append_only_and_reload(tmp_path):
    s = store.RolloutStore("x", root=tmp_path)
    s.append(_row("a"))
    with pytest.raises(FileExistsError):
        s.append(_row("a"))
    assert "a" in store.RolloutStore("x", root=tmp_path)


def test_missing_fields_refused(tmp_path):
    with pytest.raises(ValueError):
        store.RolloutStore("x", root=tmp_path).append({"rollout_id": "a"})


def test_torn_tail_is_skipped(tmp_path):
    s = store.RolloutStore("x", root=tmp_path)
    s.append(_row("a"))
    with open(s.path, "a") as f:
        f.write('{"rollout_id": "b", "trunc')  # crash mid-write
    s2 = store.RolloutStore("x", root=tmp_path)
    assert "a" in s2 and "b" not in s2
    s2.append(_row("b"))  # re-request works and lands on its own line
    assert "b" in store.RolloutStore("x", root=tmp_path)


def test_judgements_versioned_not_overwritten(tmp_path):
    j = store.JudgementStore("j", root=tmp_path)
    row = {"rollout_id": "a", "judge": "silence:x", "prompt_version": "v1", "model": "m", "output": {},
           "created_at": "", "cost_usd": 0}
    j.append(row)
    with pytest.raises(FileExistsError):
        j.append(row)
    j.append({**row, "prompt_version": "v2"})
    assert j.has("a", "silence:x", "v1", "m") and j.has("a", "silence:x", "v2", "m")
