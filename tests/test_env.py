"""The bin/td overlay must not change any package version upstream's lock pins."""
import importlib.metadata as md
import tomllib

from td import config


def _norm(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def test_overlay_keeps_locked_versions():
    lock = tomllib.loads((config.UPSTREAM_DIR / "uv.lock").read_text())
    locked: dict[str, set[str]] = {}
    for p in lock["package"]:
        if "version" in p:
            locked.setdefault(_norm(p["name"]), set()).add(p["version"])
    drift = []
    for dist in md.distributions():
        name = _norm(dist.metadata["Name"])
        if name in locked and dist.version not in locked[name]:
            drift.append(f"{name}: installed {dist.version}, locked {sorted(locked[name])}")
    assert not drift, "overlay changed locked packages:\n" + "\n".join(drift)
