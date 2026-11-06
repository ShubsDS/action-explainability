from pathlib import Path

from td import config


def test_upstream_checkout_matches_pin():
    assert config.upstream_head_sha() == config.UPSTREAM_SHA, "upstream/ HEAD differs from td.config.UPSTREAM_SHA"


def test_pin_recorded_in_notes():
    assert config.UPSTREAM_SHA in (config.ROOT / "NOTES.md").read_text()


def test_env_is_gitignored():
    lines = (config.ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in lines


def test_caps_within_total_budget():
    assert sum(config.CAPS.values()) <= config.TOTAL_BUDGET
    # Phase 1 caps from the handoff; changing them needs the user's OK.
    assert {k: config.CAPS[k] for k in ("M0", "M1", "M2", "M3", "M4", "M5a", "M5b", "M7")} == {
        "M0": 10, "M1": 15, "M2": 25, "M3": 250, "M4": 300, "M5a": 400, "M5b": 150, "M7": 275}


def test_upstream_on_path():
    import scripts.common  # noqa: F401  (resolves only if upstream/ is on sys.path)

    assert Path(scripts.common.REPO).resolve() == config.UPSTREAM_DIR.resolve()


def test_sampling_settings_match_upstream():
    import scripts.common as up

    assert up.SYS_PROMPT == config.SYS_PROMPT
    assert up.RENDERER == config.RENDERER_NAME
    assert up.BASE_MODEL == config.BASE_MODEL
