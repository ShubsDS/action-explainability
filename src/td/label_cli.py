"""Blind hand-labelling tool (for the user; Claude never labels).

A batch is `labels/<name>.batch.jsonl`: a header line {"_batch": {...labels, instructions}} and
one item per line holding ONLY whitelisted fields, so judge outputs and grades can't leak into
the view. Labels are appended to `labels/<name>.labels.jsonl` (one line per decision; the
last decision for an item wins on read). Quitting and rerunning resumes where you stopped.

    bin/td python -m td.label_cli labels/m3_silence.batch.jsonl
    bin/td python -m td.label_cli labels/m3_silence.batch.jsonl --summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

from td import config, store

M3_LABELS = ("open", "silent", "illegible", "not_a_hack")
M4_LABELS = ("admits", "denies", "unclear")
ACCIDENTAL_LABELS = ("accidental", "intentional", "unsure")

# The only fields a labeller sees. Never add judge outputs, grades, cls or detector flags here.
VIEW_FIELDS = ("user_prompt", "reasoning", "code", "answer_text", "question", "answer")


def make_batch(name: str, rows: list[dict], *, labels: tuple[str, ...], instructions: str,
               seed: int, fields: tuple[str, ...] = ("reasoning", "code")) -> Path:
    """Write a blind batch. Item order is shuffled with `seed`; item ids are opaque hashes."""
    bad = [f for f in fields if f not in VIEW_FIELDS]
    if bad:
        raise ValueError(f"fields {bad} are not viewable in a blind batch")
    path = config.LABELS_DIR / f"{name}.batch.jsonl"
    if path.exists():
        raise FileExistsError(f"{path} exists; batches are immutable once made")
    items = []
    for r in rows:
        iid = hashlib.sha256(f"{name}:{r['rollout_id']}".encode()).hexdigest()[:12]
        items.append({"item_id": iid, "rollout_id": r["rollout_id"], "task_id": r["task_id"],
                      "view": {f: r.get(f) or "" for f in fields}})
    random.Random(seed).shuffle(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"_batch": {"name": name, "labels": list(labels), "instructions": instructions,
                                       "seed": seed, "n": len(items), "created_at": store.now_iso()}}) + "\n")
        for it in items:
            f.write(json.dumps(it) + "\n")
    return path


def split_by_task(task_ids: list[str], seed: int) -> tuple[set[str], set[str]]:
    """Random 50/50 split of distinct task ids: (tune half, report half)."""
    ts = sorted(set(task_ids))
    random.Random(seed).shuffle(ts)
    h = len(ts) // 2
    return set(ts[:h]), set(ts[h:])


def read_batch(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(x) for x in open(path) if x.strip()]
    return lines[0]["_batch"], lines[1:]


def labels_path(batch_path: Path) -> Path:
    return batch_path.with_name(batch_path.name.replace(".batch.jsonl", ".labels.jsonl"))


def read_labels(batch_path: Path) -> dict[str, dict]:
    p = labels_path(batch_path)
    out: dict[str, dict] = {}
    if p.exists():
        for line in open(p):
            if line.strip():
                d = json.loads(line)
                out[d["item_id"]] = d
    return out


def _show(i: int, n: int, item: dict, meta: dict) -> None:
    os.system("clear" if os.name != "nt" else "cls")
    w = shutil.get_terminal_size((100, 40)).columns
    print(f"[{i}/{n}] item {item['item_id']}   ({meta['name']})".ljust(w, "─"))
    for k, v in item["view"].items():
        print(f"\n=== {k.upper()} ===\n{v}")
    print("\n" + "─" * w)
    print(meta["instructions"])


def run(batch_path: Path) -> None:
    meta, items = read_batch(batch_path)
    allowed = meta["labels"]
    done = read_labels(batch_path)
    todo = [it for it in items if it["item_id"] not in done]
    keys = {str(i + 1): lab for i, lab in enumerate(allowed)}
    out = labels_path(batch_path)
    for i, it in enumerate(todo, start=len(done) + 1):
        _show(i, len(items), it, meta)
        prompt = "  ".join(f"[{k}] {v}" for k, v in keys.items()) + "  [n] note+label  [q] quit > "
        while True:
            a = input(prompt).strip().lower()
            if a == "q":
                print(f"saved; {len(items) - i + 1} left")
                return
            note = ""
            if a == "n":
                note = input("note: ").strip()
                a = input("label: ").strip().lower()
            lab = keys.get(a) or (a if a in allowed else None)
            if lab:
                break
            print(f"choose one of {allowed}")
        with open(out, "a") as f:
            f.write(json.dumps({"item_id": it["item_id"], "rollout_id": it["rollout_id"], "label": lab,
                                "note": note, "labeller": os.environ.get("USER", "?"),
                                "created_at": store.now_iso()}) + "\n")
            f.flush()
            os.fsync(f.fileno())
    print("batch complete")


def summary(batch_path: Path) -> None:
    meta, items = read_batch(batch_path)
    done = read_labels(batch_path)
    counts: dict[str, int] = {}
    for d in done.values():
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    print(f"{meta['name']}: {len(done)}/{len(items)} labelled; {counts}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch", type=Path)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    (summary if a.summary else run)(a.batch)


if __name__ == "__main__":
    main(sys.argv[1:])
