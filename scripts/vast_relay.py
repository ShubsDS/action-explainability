"""Runs ON the rented machine, next to vLLM. A job queue, so that neither a long sample nor the
whole run depends on our laptop staying connected (or awake).

    POST /jobs        {"id": "...", "body": {...the /v1/completions body...}}   -> {"status": ...}
    GET  /state                                         -> {"done": [ids], "active": [ids]}
    GET  /jobs/<id>                                     -> {"status": "done"|"error"|"running"|"unknown", ...}

Every sample is submitted up front. The relay runs at most RELAY_CONCURRENCY of them against vLLM
on 127.0.0.1:8000 at a time and keeps each finished result on disk; the client only makes short
calls. Submitting an id twice never runs it twice, so a client that restarts picks its jobs up
again. A failed job is reported once and forgotten, so it can be submitted again.

Spending guards, enforced here because the laptop may be asleep: the machine stops ITSELF (with
Vast's machine-scoped key, which can only act on this machine) at STOP_AT (unix time), or when
the queue has been empty for IDLE_STOP_MIN minutes after at least one job ran. Results stay on
the stopped machine's disk. The deadline can be moved while running by writing a unix time to
$WORK/stop_at (read every minute), e.g. when the spending cap is raised.

    WORK=/workspace/td STOP_AT=... /workspace/td/venv-vllm/bin/python vast_relay.py
"""
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

WORK = Path(os.environ.get("WORK", "/workspace/td"))
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
KEY = os.environ.get("VLLM_API_KEY") or (WORK / ".vllm_key").read_text().strip()
VLLM = "http://127.0.0.1:8000"
CONCURRENCY = int(os.environ.get("RELAY_CONCURRENCY", "16"))
STOP_AT = float(os.environ.get("STOP_AT", "0"))            # 0: no deadline
IDLE_STOP_MIN = float(os.environ.get("IDLE_STOP_MIN", "45"))
RETRIES = 3                               # more tries for a job whose connection to vLLM drops

app = FastAPI()
active: dict[str, asyncio.Task] = {}      # queued or running
failed: dict[str, str] = {}               # reported once, then forgotten
stats = {"started": time.time(), "finished": 0, "errors": 0, "retries": 0, "tokens": 0, "last_activity": time.time()}
sem: asyncio.Semaphore | None = None
http = httpx.AsyncClient(base_url=VLLM, headers={"Authorization": f"Bearer {KEY}"},
                         timeout=httpx.Timeout(connect=30, read=3600, write=120, pool=None),
                         limits=httpx.Limits(max_connections=None))


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S", time.gmtime()), msg, flush=True)


def _path(job_id: str) -> Path:
    if not job_id.isalnum() or len(job_id) > 64:
        raise HTTPException(400, "bad id")
    return RESULTS / f"{job_id}.json"


def _auth(request: Request) -> None:
    if request.headers.get("authorization") != f"Bearer {KEY}":
        raise HTTPException(401, "unauthorized")


async def _stream_once(body: dict) -> tuple[dict, str | None]:
    """One streamed completion from vLLM: (what came back, an error or None)."""
    out = {"token_ids": [], "logprob_tokens": [], "finish_reason": None, "stop_reason": None,
           "completion_tokens": None}
    error = None
    async with http.stream("POST", "/v1/completions", json={**body, "stream": True,
                                                            "stream_options": {"include_usage": True}}) as resp:
        if resp.status_code != 200:
            error = f"vLLM answered {resp.status_code}: {(await resp.aread()).decode('utf-8', 'replace')[:300]}"
        else:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    error = f"vLLM stream error: {str(chunk['error'])[:300]}"
                    break
                if chunk.get("usage"):
                    out["completion_tokens"] = chunk["usage"].get("completion_tokens")
                for ch in chunk.get("choices") or []:
                    out["token_ids"].extend(ch.get("token_ids") or [])
                    out["logprob_tokens"].extend((ch.get("logprobs") or {}).get("tokens") or [])
                    if ch.get("finish_reason"):
                        out["finish_reason"], out["stop_reason"] = ch["finish_reason"], ch.get("stop_reason")
    if error is None and out["finish_reason"] not in ("stop", "length"):
        error = f"stream ended without a finish_reason ({out['finish_reason']!r})"
    return out, error


async def _run(job_id: str, body: dict) -> None:
    """Run one job against vLLM; store the token ids and how it ended, or note the error.

    A connection that drops is tried again, from the start, up to RETRIES more times: in the
    scoring-fact run every request vLLM refused closed a connection that the next job then reused,
    and 46 jobs died of that with nothing wrong with them (NOTES.md, 2026-11-21). A sample is kept
    whole or not at all, so the tokens of a dropped stream are thrown away. An answer from vLLM
    that is an error (a refused request) is not tried again."""
    out, error = {}, None
    async with sem:
        for attempt in range(RETRIES + 1):
            try:
                out, error = await _stream_once(body)
                break
            except httpx.TransportError as e:
                error = f"{type(e).__name__}: {str(e)[:300]}"
                if attempt < RETRIES:
                    stats["retries"] += 1
                    log(f"job {job_id[:8]} lost its connection ({error[:80]}); trying again ({attempt + 1} of {RETRIES})")
                    await asyncio.sleep(2 * (attempt + 1))
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)[:300]}"
                break
    if error is None:
        tmp = _path(job_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(out))
        tmp.rename(_path(job_id))
        stats["finished"] += 1
        stats["tokens"] += max(len(out["token_ids"]), len(out["logprob_tokens"]))
    else:
        failed[job_id] = error
        stats["errors"] += 1
        log(f"job {job_id[:8]} failed: {error[:400]}")
    active.pop(job_id, None)
    stats["last_activity"] = time.time()


@app.post("/jobs")
async def submit(request: Request):
    _auth(request)
    req = await request.json()
    job_id, body = req["id"], req["body"]
    if _path(job_id).exists():
        return {"status": "done"}
    if job_id not in active:
        failed.pop(job_id, None)
        active[job_id] = asyncio.create_task(_run(job_id, body))
        stats["last_activity"] = time.time()
    return {"status": "running"}


@app.get("/state")
async def state(request: Request):
    _auth(request)
    return {"done": [p.stem for p in RESULTS.glob("*.json")], "active": list(active), "failed": list(failed)}


@app.get("/jobs/{job_id}")
async def result(job_id: str, request: Request):
    _auth(request)
    p = _path(job_id)
    if p.exists():
        return {"status": "done", "result": json.loads(p.read_text())}
    if job_id in failed:
        return {"status": "error", "error": failed.pop(job_id)}
    return {"status": "running" if job_id in active else "unknown"}


@app.get("/health")
async def health():
    up = time.time() - stats["started"]
    return {"active": len(active), "done_on_disk": len(list(RESULTS.glob("*.json"))), "finished": stats["finished"],
            "errors": stats["errors"], "retries": stats["retries"], "tokens": stats["tokens"], "tokens_per_s": round(stats["tokens"] / max(up, 1), 1),
            "stop_at": STOP_AT, "idle_min": round((time.time() - stats["last_activity"]) / 60, 1)}


def _stop_machine(reason: str) -> None:
    env = dict(e.split("=", 1) for e in open("/proc/1/environ").read().split("\0") if "=" in e)
    log(f"STOPPING THE MACHINE: {reason}")
    r = subprocess.run([str(WORK / "venv-vllm/bin/vastai"), "stop", "instance", env["CONTAINER_ID"],
                        "--api-key", env["CONTAINER_API_KEY"]], capture_output=True, text=True)
    log(f"stop command rc {r.returncode}: {(r.stdout + r.stderr)[:200].replace(env['CONTAINER_API_KEY'], '<key>')}")


async def _guard() -> None:
    while True:
        await asyncio.sleep(60)
        now = time.time()
        global STOP_AT
        try:
            new = float((WORK / "stop_at").read_text().strip())
            if new != STOP_AT:
                log(f"deadline moved to {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(new))}")
                STOP_AT = new
        except (FileNotFoundError, ValueError):
            pass
        if STOP_AT and now >= STOP_AT:
            _stop_machine(f"deadline reached with {len(active)} jobs unfinished")
        elif stats["finished"] + stats["errors"] > 0 and not active and now - stats["last_activity"] > IDLE_STOP_MIN * 60:
            _stop_machine(f"queue empty for {IDLE_STOP_MIN:.0f} min")
        else:
            continue
        await asyncio.sleep(600)   # if the stop did not take, try again in 10 minutes


@app.on_event("startup")
async def _startup() -> None:
    global sem
    sem = asyncio.Semaphore(CONCURRENCY)
    asyncio.create_task(_guard())
    log(f"relay up: concurrency {CONCURRENCY}, stop_at {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(STOP_AT)) if STOP_AT else 'none'}, "
        f"idle stop {IDLE_STOP_MIN:.0f} min, {len(list(RESULTS.glob('*.json')))} results on disk")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
