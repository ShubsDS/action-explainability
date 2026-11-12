import asyncio

import pytest

from td import config, render


def test_fork_k0_empty_insert_is_parent_prompt():
    parent = {"prompt_tokens": [1, 2, 3], "output_tokens": [4, 5, 6, 7]}
    prefix, budget = render.fork_prefix(parent, 0)
    assert prefix == [1, 2, 3] and budget == config.MAX_TOKENS - 3  # 65,536 is total context


def test_fork_keeps_first_k_output_tokens_and_insert():
    parent = {"prompt_tokens": [1, 2, 3], "output_tokens": [4, 5, 6, 7]}
    prefix, budget = render.fork_prefix(parent, 2, [99, 98])
    assert prefix == [1, 2, 3, 4, 5, 99, 98]
    assert budget == config.MAX_TOKENS - 7


def test_fork_out_of_range():
    with pytest.raises(ValueError):
        render.fork_prefix({"prompt_tokens": [1], "output_tokens": [2]}, 5)


@pytest.mark.network  # tokenizer + dataset download
def test_single_turn_prompt_matches_upstream_env():
    """M2 acceptance test 1: our prompt == RewardHackEnv.initial_observation() as heldout_gen builds it."""
    from src.train.rlaif.reward_hack.env import RewardHackEnv

    from td import tasks

    item = tasks.load(("conflicting",), held_out=True)[0]
    ours, _, user_prompt = asyncio.run(render.single_turn_prompt(item))
    env = RewardHackEnv(renderer=render.renderer(), sys_prompt="You are Corin.", timeout=8,
                        show_test_in_prompt=True, **item)
    obs, _ = asyncio.run(env.initial_observation())
    assert ours == list(obs.to_ints())
    assert item["test"] in user_prompt


def test_multi_turn_not_built_yet():
    with pytest.raises(NotImplementedError):
        render.multi_turn_context({}, [], keep_reasoning=True)
