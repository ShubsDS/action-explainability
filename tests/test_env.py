"""The bin/td overlay must not change any package version upstream's lock pins, apart from the
deliberate overrides listed here (each explained in NOTES.md)."""
import importlib.metadata as md
import tomllib

from td import config

# package -> the exact version we override it to
ALLOWED_OVERRIDES = {"tinker": "0.39.0"}  # 0.25.0 rejected by the Tinker server, 2026-11-11


def _norm(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def test_overlay_keeps_locked_versions():
    lock = tomllib.loads((config.UPSTREAM_DIR / "uv.lock").read_text())
    locked: dict[str, set[str]] = {}
    for p in lock["package"]:
        if "version" in p:
            locked.setdefault(_norm(p["name"]), set()).add(p["version"])
    drift = []
    # Check only the copy that wins on sys.path (md.distribution resolves first-on-path); the
    # base venv's shadowed copy of an overridden package is still listed by md.distributions().
    names = {dist.metadata["Name"] for dist in md.distributions()}
    for raw in sorted(names):
        dist = md.distribution(raw)
        name = _norm(raw)
        if name in ALLOWED_OVERRIDES:
            if dist.version != ALLOWED_OVERRIDES[name]:
                drift.append(f"{name}: installed {dist.version}, override pins {ALLOWED_OVERRIDES[name]}")
        elif name in locked and dist.version not in locked[name]:
            drift.append(f"{name}: installed {dist.version}, locked {sorted(locked[name])}")
    assert not drift, "overlay changed locked packages:\n" + "\n".join(drift)


def test_overrides_match_requirements_extra():
    pins = {}
    for line in (config.ROOT / "requirements-extra.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            n, v = line.split("==")
            pins[_norm(n)] = v
    assert {k: pins.get(k) for k in ALLOWED_OVERRIDES} == ALLOWED_OVERRIDES
