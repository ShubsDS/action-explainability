import pytest

from td import tasks

pytestmark = pytest.mark.network  # HF dataset download (cached after the first run)


def test_split_sizes():
    assert len(tasks.load(("conflicting",), held_out=True)) == tasks.N_HELDOUT
    assert len(tasks.load(("conflicting",), held_out=False)) == tasks.N_TRAIN


def test_broken_scan_matches_known():
    h = tasks.broken_task_ids(tasks.load(("conflicting",), held_out=True))
    t = tasks.broken_task_ids(tasks.load(("conflicting",), held_out=False))
    assert set(h) == tasks.KNOWN_BROKEN_HELDOUT
    assert set(t) == tasks.KNOWN_BROKEN_TRAIN
    assert len(tasks.usable(tasks.load(("conflicting",), held_out=True))) == 20


def test_no_original_test_broken():
    items = tasks.load(("original",), held_out=True) + tasks.load(("original",), held_out=False)
    assert tasks.broken_task_ids(items) == {}


def test_order_is_upstream_seeded_shuffle():
    a = [i["task_id"] for i in tasks.load(("conflicting",), held_out=True)]
    b = [i["task_id"] for i in tasks.load(("conflicting",), held_out=True)]
    assert a == b


def test_broken_reason_kinds():
    assert tasks.broken_reason("assert f(1) == 2\n") == "no_check"
    assert tasks.broken_reason('def check(c):\n    assert c("x) == 1\n') == "syntax"
    assert tasks.broken_reason("def check(c):\n    assert c(1) == 1\n") is None
