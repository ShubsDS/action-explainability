"""Rebuild a released checkpoint in our own Tinker account from upstream's Hugging Face mirror.

Why: on 2026-11-16 28 of upstream's 30 public Tinker sampler paths stopped resolving (NOTES.md).
The Hugging Face mirrors are the PEFT adapters Tinker itself exported from those paths, so the
same weights can be uploaded back as "external weights" and saved for sampling.

    1. download adapter_config.json and adapter_model.safetensors from the checkpoint's hf_repo
       into an empty directory (and check the safetensors sha256 against the Hub's)
    2. bin/td python -m td.rebuild_checkpoint anti 1 30 --adapter-dir DIR --sha256 <hub sha256>

Steps, each recorded in DIR/rebuild_state.json so a rerun resumes: start the upload, PUT the files,
finish (-> an external-weights path), load it into a training client of the same LoRA config,
save sampler weights (-> a sampler path), and write the entry to rebuilt_checkpoints.json.
`td.checkpoints` then samples from that path while the checkpoint keeps its published identity.

Nothing here samples or trains. The rebuilt copy must still be checked against stored behaviour
before it is used (see NOTES.md).

NOT YET WORKING (2026-11-16): `external_weights_upload_start` answers 400 "Uploading external
weights is not enabled for this organization". The feature is experimental and switched on per
organization by Tinker; ask their support. Everything after that call is untested.

Second risk, from the SDK's own docstring: the files "must be exactly what `save_weights_external`
exports for `base_model` and `lora_config`; finishing rejects anything else". The Hugging Face
mirror was made by Tinker's own export tool, so it should match, but that is unproven. The upload's
run "cannot be trained on", so the weights are loaded into a fresh training client with
`load_state`, as the SDK's upload docs say.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

from td import checkpoints, config


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _put(path: Path, url: str) -> None:
    """HTTP PUT of the file's bytes to a signed URL. The URL is a capability: never printed."""
    r = subprocess.run(["curl", "--fail", "-s", "-S", "--retry", "3", "-T", str(path), url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"upload of {path.name} failed: {r.stderr.strip()[:300].replace(url, '<signed url>')}")


def rebuild(character: str, seed: int | None, step: int, adapter_dir: Path, expected_sha: str,
            rank: int, train_unembed: bool, train_mlp: bool, train_attn: bool) -> dict:
    import tinker
    from tinker import types

    ck = checkpoints.get(character, seed, step)
    state_path = adapter_dir / "rebuild_state.json"
    st = json.loads(state_path.read_text()) if state_path.exists() else {}

    def save(**kw):
        st.update(kw)
        state_path.write_text(json.dumps(st, indent=1))

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg["base_model_name_or_path"] != config.BASE_MODEL or cfg["r"] != rank:
        raise RuntimeError(f"adapter is for {cfg['base_model_name_or_path']} rank {cfg['r']}, expected {config.BASE_MODEL} rank {rank}")
    if "sha256" not in st:
        got = sha256(adapter_dir / "adapter_model.safetensors")
        if got != expected_sha:
            raise RuntimeError(f"adapter_model.safetensors sha256 {got} != expected {expected_sha}")
        save(sha256=got)
        print("sha256 matches the Hub's")

    service = tinker.ServiceClient()
    rest = service.create_rest_client()
    lora = types.LoraConfig(rank=rank, train_unembed=train_unembed, train_mlp=train_mlp, train_attn=train_attn)

    if "external_path" not in st:
        start = rest.external_weights_upload_start(config.BASE_MODEL, lora).result()
        names = sorted(start.urls)
        print("upload started; files wanted:", names, "| urls expire", start.expires)
        missing = [n for n in names if not (adapter_dir / n).exists()]
        if missing:
            raise RuntimeError(f"the upload wants files we do not have: {missing}")
        for n in names:
            _put(adapter_dir / n, start.urls[n])
            print("  uploaded", n)
        fin = rest.external_weights_upload_finish(start.upload_id).result()
        save(external_path=fin.tinker_path, uploaded_files=names)
    print("external weights:", st["external_path"])

    if "served_path" not in st:
        # The SDK's upload docs: "Load the checkpoint into a training client of the same base model
        # and LoRA config with `load_state` (weights only)." The upload's own run cannot be trained on.
        tc = service.create_lora_training_client(
            config.BASE_MODEL, rank=rank, train_mlp=train_mlp, train_attn=train_attn, train_unembed=train_unembed,
            user_metadata={"rebuilt_from": ck.hf_repo, "label": ck.label}).result()
        tc.load_state(st["external_path"]).result()
        name = f"{ck.label.replace('@', '-step')}-rebuilt"
        res = tc.save_weights_for_sampler(name).result()   # no ttl: never expires
        save(served_path=res.path)
    print("sampler weights:", st["served_path"])

    info = rest.get_weights_info_by_tinker_path(st["served_path"]).result()
    entry = {"published_path": ck.sampler_path, "hf_repo": ck.hf_repo, "served_path": st["served_path"],
             "external_path": st["external_path"], "adapter_sha256": st["sha256"],
             "lora": {"rank": info.lora_rank, "train_unembed": info.train_unembed, "train_mlp": info.train_mlp,
                      "train_attn": info.train_attn},
             "rebuilt_on": datetime.date.today().isoformat()}
    if info.base_model != config.BASE_MODEL or info.lora_rank != rank:
        raise RuntimeError(f"rebuilt weights report {info.base_model} rank {info.lora_rank}")
    table = checkpoints.rebuilt()
    table[ck.label] = entry
    checkpoints.REBUILT_PATH.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    print(f"wrote {checkpoints.REBUILT_PATH.name}: {ck.label}")
    return entry


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("character")
    ap.add_argument("seed", type=int, help="RL seed (ignored for step 0)")
    ap.add_argument("step", type=int)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--sha256", required=True, help="the Hub's sha256 of adapter_model.safetensors")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--train-unembed", action="store_true", help="upstream's adapters do not train the unembedding")
    a = ap.parse_args()
    rebuild(a.character, a.seed, a.step, a.adapter_dir, a.sha256, a.rank, a.train_unembed, True, True)
