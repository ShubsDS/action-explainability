"""Sampling from a vLLM server we run ourselves, when Tinker cannot serve a checkpoint.

Why: on 2026-11-16 the published Tinker checkpoints stopped resolving and Tinker would not let us
upload the Hugging Face copies (NOTES.md). The same adapters, converted with
`tinker_cookbook.weights.build_lora_adapter`, load into vLLM next to the base model
(`scripts/vast_serve.sh`). A checkpoint whose entry in `rebuilt_checkpoints.json` has a
`served_path` of `vllm://<adapter name>` is sampled here instead of from Tinker.

Same request as the Tinker path: token ids in, token ids out, temperature 1.0, no top-p or top-k
cut, stop at the renderer's stop token, and the stop token kept as the last sampled token.

    TD_VLLM_URL       server address, default http://127.0.0.1:8000 (reach it through an SSH tunnel)
    TD_VLLM_RELAY_URL address of `scripts/vast_relay.py` on the machine, e.g. http://127.0.0.1:8001.
                      When set, a sample is submitted as a job (its id is the rollout id) and polled
                      for. The machine works through the queue by itself, so a dropped tunnel or a
                      sleeping laptop loses nothing, and a restarted run picks its jobs up again.
                      Without it the answer is streamed over one long connection.
    TD_VLLM_API_KEY   the key the server was started with (in .env; never printed)

    bin/td python -m td.vllm_client register anti 1 30 --name anti_s1_step30 --sha256 <hub sha256>
    bin/td python -m td.vllm_client smoke anti 1 30        # one short request; costs machine time only
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import random
import uuid
from dataclasses import dataclass

from td import checkpoints, config

SCHEME = "vllm://"
DEFAULT_URL = "http://127.0.0.1:8000"


def is_vllm(model_path: str | None) -> bool:
    return bool(model_path) and model_path.startswith(SCHEME)


@dataclass
class Sampled:
    """What `Sampler` needs from one sample; the same two fields the Tinker completer returns."""
    tokens: list[int]
    stop_reason: str   # "stop" | "length"


class VllmError(RuntimeError):
    pass


def _ids_from_logprob_tokens(tokens: list[str]) -> list[int]:
    """`return_tokens_as_token_ids` spells each sampled token as "token_id:123"."""
    out = []
    for t in tokens:
        if not t.startswith("token_id:"):
            raise VllmError(f"expected 'token_id:N' in logprobs, got {t!r}")
        out.append(int(t[len("token_id:"):]))
    return out


class VllmClient:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, transport=None):
        self.model = model
        self.base_url = (base_url or os.environ.get("TD_VLLM_URL") or DEFAULT_URL).rstrip("/")
        self._api_key = api_key if api_key is not None else os.environ.get("TD_VLLM_API_KEY", "")
        self._transport = transport   # injectable for tests
        self._http = None
        self.relay_url = (os.environ.get("TD_VLLM_RELAY_URL") or "").rstrip("/") or None
        self.poll_seconds = 20.0
        self.give_up_seconds = 1800.0   # of continuous failure to reach the relay
        self._state = None              # (fetched at, {"done": set, "active": set}); shared by all samples
        self._state_lock = None

    @classmethod
    def from_path(cls, model_path: str) -> "VllmClient":
        if not is_vllm(model_path):
            raise ValueError(f"not a vllm path: {model_path!r}")
        return cls(model_path[len(SCHEME):])

    def http(self):
        if self._http is None:
            import httpx

            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            # A 65,536-token answer streams for an hour or more; only a silent socket is an error.
            self._http = httpx.AsyncClient(
                base_url=self.base_url, headers=headers, transport=self._transport,
                timeout=httpx.Timeout(connect=30, read=1800, write=120, pool=None),
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=8))
        return self._http

    def request_body(self, prompt_tokens: list[int], max_tokens: int, temperature: float,
                     stop_token_ids: list[int]) -> dict:
        return {
            "model": self.model, "prompt": list(prompt_tokens), "max_tokens": max_tokens,
            # Tinker's SamplingParams defaults: temperature as given, top_p 1, top_k off.
            "temperature": temperature, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0,
            "stop_token_ids": list(stop_token_ids),
            "add_special_tokens": False, "skip_special_tokens": False,
            # Two ways to get the sampled ids back; whichever the server supports is used, and
            # if both come back they must agree.
            "return_token_ids": True, "logprobs": 1, "return_tokens_as_token_ids": True,
            "stream": True, "stream_options": {"include_usage": True},
        }

    async def sample(self, prompt_tokens: list[int], max_tokens: int, temperature: float,
                     stop_token_ids: list[int], job_id: str | None = None) -> Sampled:
        """`job_id` (relay only): a stable id for this sample. Asking again with the same id
        returns the same sample instead of drawing a second one."""
        if not all(isinstance(s, int) for s in stop_token_ids):
            raise VllmError(f"only token-id stop conditions are supported, got {stop_token_ids!r}")
        body = self.request_body(prompt_tokens, max_tokens, temperature, stop_token_ids)
        got = await (self._via_relay(body, job_id) if self.relay_url else self._via_stream(body))
        return _check(got, stop_token_ids, max_tokens)

    async def _relay_state(self) -> tuple[float, dict]:
        """Which jobs the relay has finished or still holds: one short call shared by every
        waiting sample, refreshed at most every half poll interval."""
        loop = asyncio.get_running_loop()
        if self._state_lock is None:
            self._state_lock = asyncio.Lock()
        async with self._state_lock:
            if self._state is None or loop.time() - self._state[0] > self.poll_seconds / 2:
                asked = loop.time()
                r = await self.http().get(f"{self.relay_url}/state", timeout=60)
                r.raise_for_status()
                d = r.json()
                self._state = (asked, {"done": set(d["done"]), "active": set(d["active"]) | set(d.get("failed", []))})
            return self._state

    async def _via_stream(self, body: dict) -> dict:
        """One streamed request; the connection must stay up until the sample ends."""
        got = {"token_ids": [], "logprob_tokens": [], "finish_reason": None, "stop_reason": None,
               "completion_tokens": None}
        async with self.http().stream("POST", "/v1/completions", json=body) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode("utf-8", "replace")
                raise VllmError(f"vLLM answered {resp.status_code}: {text[:300]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise VllmError(f"vLLM stream error: {str(chunk['error'])[:300]}")
                if chunk.get("usage"):
                    got["completion_tokens"] = chunk["usage"].get("completion_tokens")
                for ch in chunk.get("choices") or []:
                    got["token_ids"].extend(ch.get("token_ids") or [])
                    got["logprob_tokens"].extend((ch.get("logprobs") or {}).get("tokens") or [])
                    if ch.get("finish_reason"):
                        got["finish_reason"], got["stop_reason"] = ch["finish_reason"], ch.get("stop_reason")
        return got

    async def _via_relay(self, body: dict, job_id: str | None = None) -> dict:
        """Submit the request as a job on the machine and wait for it with short calls. A call that
        fails (tunnel down) is simply retried; the job id makes a repeated submit harmless."""
        import httpx

        job, url = job_id or uuid.uuid4().hex, self.relay_url
        failing_since = submitted_at = None
        loop = asyncio.get_running_loop()
        while True:
            try:
                if submitted_at is None:
                    r = await self.http().post(f"{url}/jobs", json={"id": job, "body": body}, timeout=60)
                    r.raise_for_status()
                    submitted_at = loop.time()
                else:
                    asked, st = await self._relay_state()
                    if asked > submitted_at:           # a state from before the submit says nothing
                        if job in st["done"] or job not in st["active"]:
                            r = await self.http().get(f"{url}/jobs/{job}", timeout=120)
                            r.raise_for_status()
                            ans = r.json()
                            if ans["status"] == "done":
                                return ans["result"]
                            if ans["status"] == "error":
                                raise VllmError(ans["error"])
                            if ans["status"] == "unknown":
                                raise VllmError("the relay lost the job (it restarted); the sample is discarded")
                failing_since = None
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                failing_since = failing_since or loop.time()
                if loop.time() - failing_since > self.give_up_seconds:
                    raise VllmError(f"relay unreachable for {self.give_up_seconds:.0f} s: {type(e).__name__}") from e
            await asyncio.sleep(self.poll_seconds * random.uniform(0.75, 1.25))

    async def served_models(self) -> list[str]:
        r = await self.http().get("/v1/models")
        r.raise_for_status()
        return [m["id"] for m in r.json()["data"]]


def _check(got: dict, stop_token_ids: list[int], max_tokens: int) -> Sampled:
    """Turn what the server sent into a sample, or refuse it."""
    direct = list(got.get("token_ids") or [])
    via_lp = _ids_from_logprob_tokens(got.get("logprob_tokens") or [])
    finish, stop_tok, n_completion = got.get("finish_reason"), got.get("stop_reason"), got.get("completion_tokens")
    if finish not in ("stop", "length"):
        raise VllmError(f"stream ended without a usable finish_reason ({finish!r}); the sample is discarded")
    if direct and via_lp and direct != via_lp:
        raise VllmError("the server's token_ids and logprob tokens disagree; the sample is discarded")
    tokens = direct or via_lp
    if not tokens:
        raise VllmError("the server returned no token ids (needs return_token_ids or return_tokens_as_token_ids)")
    if finish == "stop" and tokens[-1] not in stop_token_ids:
        # vLLM drops the token that ended the sample; Tinker keeps it, and the parser expects it.
        if isinstance(stop_tok, int) and stop_tok in stop_token_ids:
            tokens = tokens + [stop_tok]
        elif len(stop_token_ids) == 1:
            tokens = tokens + [stop_token_ids[0]]   # end-of-sequence: the same token for this model
        else:
            raise VllmError(f"cannot tell which stop token ended the sample (stop_reason {stop_tok!r})")
    if n_completion is not None and abs(len(tokens) - n_completion) > 1:
        raise VllmError(f"got {len(tokens)} token ids but the server counted {n_completion}; the sample is discarded")
    if len(tokens) > max_tokens + 1:
        raise VllmError(f"{len(tokens)} tokens exceeds max_tokens {max_tokens}")
    return Sampled(tokens=tokens, stop_reason=finish)


# ---------------------------------------------------------------- registration and a smoke test

def register(character: str, seed: int | None, step: int, name: str, adapter_sha256: str, note: str) -> dict:
    """Point a released checkpoint at an adapter served by our vLLM server. Identity is unchanged:
    cache keys and stored rows keep the published path; each new row records `served_path`."""
    ck = checkpoints.get(character, seed, step)   # sampler_path and hf_repo are always the published ones
    table = checkpoints.rebuilt()
    table[ck.label] = {"published_path": ck.sampler_path, "hf_repo": ck.hf_repo, "served_path": SCHEME + name,
                       "backend": "vllm", "adapter_sha256": adapter_sha256, "note": note,
                       "rebuilt_on": datetime.date.today().isoformat()}
    checkpoints.REBUILT_PATH.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    return table[ck.label]


async def smoke(character: str, seed: int | None, step: int) -> None:
    """One 64-token request with a real task prompt. Shows what the server returns, so the first
    paid minutes tell us whether token ids, the stop token and the adapter name all work."""
    from td import render
    from td.experiments import m5_reasoning as m5

    ck = checkpoints.get(character, seed, step)
    if not is_vllm(ck.model_path):
        raise SystemExit(f"{ck.label} is not registered for vLLM (model_path {ck.model_path!r})")
    c = VllmClient.from_path(ck.model_path)
    print("server:", c.base_url, "| serves:", await c.served_models(), "| asking for:", c.model)
    items = m5.usable_items()
    prompt, _, _ = await render.single_turn_prompt(items[sorted(items)[0]])
    rend = render.renderer()
    res = await c.sample(prompt, 64, config.TEMPERATURE, rend.get_stop_sequences())
    print(f"prompt {len(prompt)} tokens -> {len(res.tokens)} sampled, stop_reason {res.stop_reason}")
    print("text:", repr(rend.tokenizer.decode(res.tokens))[:300])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("register", "smoke"):
        p = sub.add_parser(cmd)
        p.add_argument("character")
        p.add_argument("seed", type=int, help="RL seed (ignored for step 0)")
        p.add_argument("step", type=int)
        if cmd == "register":
            p.add_argument("--name", required=True, help="the adapter's name on the vLLM server (--lora-modules NAME=...)")
            p.add_argument("--sha256", required=True, help="the Hub's sha256 of the adapter the server converted")
            p.add_argument("--note", default="", help="where it runs, e.g. 'vast 4xA100, vllm 0.18.1, BF16'")
    a = ap.parse_args()
    if a.cmd == "register":
        print(json.dumps(register(a.character, a.seed, a.step, a.name, a.sha256, a.note), indent=1))
    else:
        asyncio.run(smoke(a.character, a.seed, a.step))
