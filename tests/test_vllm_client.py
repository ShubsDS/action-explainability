"""The self-hosted vLLM path: same request as Tinker, token ids back, stop token kept, $0 per
sample, and a rented machine as one ledger entry. No network: httpx is given a fake transport."""
import asyncio
import json

import httpx
import pytest

from td import checkpoints as ck
from td import config, cost, grading, render, rental, sampling, store, vllm_client
from td.vllm_client import VllmClient, VllmError

STOP = 11


def _sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def _client(chunks, seen=None, status=200):
    def handler(request):
        if seen is not None:
            seen.append((request.url.path, request.headers.get("authorization"), json.loads(request.content)))
        return httpx.Response(status, text=_sse(chunks) if status == 200 else "boom")
    return VllmClient("anti_s1_step30", base_url="http://x", api_key="k", transport=httpx.MockTransport(handler))


def _chunk(ids=None, lp=None, finish=None, stop_reason=None):
    c = {"finish_reason": finish, "stop_reason": stop_reason}
    if ids is not None:
        c["token_ids"] = ids
    if lp is not None:
        c["logprobs"] = {"tokens": [f"token_id:{i}" for i in lp]}
    return {"choices": [c]}


def run(c, max_tokens=100):
    return asyncio.run(c.sample([1, 2, 3], max_tokens, 1.0, [STOP]))


def test_request_matches_the_tinker_settings():
    seen = []
    run(_client([_chunk(ids=[5, 6]), _chunk(ids=[STOP], finish="stop", stop_reason=STOP)], seen))
    path, auth, body = seen[0]
    assert path == "/v1/completions" and auth == "Bearer k"
    assert body["model"] == "anti_s1_step30" and body["prompt"] == [1, 2, 3] and body["max_tokens"] == 100
    assert (body["temperature"], body["top_p"], body["top_k"], body["repetition_penalty"]) == (1.0, 1.0, -1, 1.0)
    assert body["stop_token_ids"] == [STOP] and body["skip_special_tokens"] is False and body["add_special_tokens"] is False


def test_stop_token_is_kept_whether_or_not_the_server_returns_it():
    with_it = run(_client([_chunk(ids=[5, 6]), _chunk(ids=[STOP], finish="stop", stop_reason=STOP)]))
    without = run(_client([_chunk(ids=[5, 6]), _chunk(ids=[], finish="stop", stop_reason=None)]))
    assert with_it.tokens == without.tokens == [5, 6, STOP]
    assert with_it.stop_reason == without.stop_reason == "stop"


def test_length_stop_adds_nothing():
    r = run(_client([_chunk(ids=[5, 6, 7], finish="length")]), max_tokens=3)
    assert r.tokens == [5, 6, 7] and r.stop_reason == "length"


def test_ids_can_come_from_logprob_tokens_alone():
    r = run(_client([_chunk(lp=[5, 6]), _chunk(lp=[STOP], finish="stop")]))
    assert r.tokens == [5, 6, STOP]


@pytest.mark.parametrize("chunks,why", [
    ([_chunk(ids=[5, 6], lp=[5, 9], finish="length")], "disagree"),
    ([_chunk(ids=[5, 6])], "finish_reason"),                       # stream cut off
    ([_chunk(finish="stop")], "no token ids"),
    ([_chunk(ids=[5, 6], finish="length"), {"choices": [], "usage": {"completion_tokens": 40}}], "counted"),
])
def test_a_sample_we_cannot_trust_is_refused(chunks, why):
    with pytest.raises(VllmError, match=why):
        run(_client(chunks))


def test_http_error_is_reported():
    with pytest.raises(VllmError, match="500"):
        run(_client([], status=500))


def test_self_hosted_samples_cost_nothing_and_record_where_they_ran(tmp_data, monkeypatch):
    monkeypatch.setattr(grading, "require_docker", lambda *a, **k: None)
    monkeypatch.setattr(render, "renderer", lambda: None)
    c = ck.Checkpoint("anti", 1, 30, "tinker://fake/sampler_weights/000030", None, "hf/x", "vllm://anti_s1_step30")
    plain = ck.Checkpoint("anti", 1, 30, "tinker://fake/sampler_weights/000030", None, "hf/x")
    s = sampling.Sampler(c, cost.Budget("M2"), "vllm")
    assert s.self_hosted and not sampling.Sampler(plain, cost.Budget("M2"), "t").self_hosted
    assert isinstance(s.client(), VllmClient) and s.client().model == "anti_s1_step30"

    jobs, asked = [], []

    async def fake_sample(self, prompt_tokens, max_tokens, job_id=None):
        jobs.append(job_id)
        asked.append(max_tokens)
        return vllm_client.Sampled(tokens=[7] * 50 + [STOP], stop_reason="stop")

    monkeypatch.setattr(sampling.Sampler, "_sample", fake_sample)
    spec = sampling.SampleSpec(item={"task_id": "t0", "impossible_type": "conflicting"}, sample_idx=0, prompt_tokens=[1, 2, 3])
    key = s.key(spec)
    raw = asyncio.run(s._pay_and_journal(spec, key, store.rollout_id(key)))
    assert raw["cost_usd"] == 0.0 and raw["served_path"] == "vllm://anti_s1_step30"
    assert jobs == [store.rollout_id(key)]                 # the job is named after the rollout
    assert asked == [config.MAX_TOKENS - 3]                # prompt + output never passes Tinker's total
    assert key == sampling.spec_key(plain, spec)          # same identity as the published checkpoint
    st = cost.Budget("M2").status()
    assert st["settled"] == 0 and st["outstanding"] == 0


def test_our_own_server_is_asked_for_what_is_left_of_the_total():
    """Tinker stops at prompt + output = 65,536; asking vLLM for 65,536 more than the prompt let
    answers run past that and got long prompts refused (NOTES.md, 2026-11-21)."""
    room = sampling.self_hosted_room
    assert room([0] * 1_000, config.MAX_TOKENS) == config.MAX_TOKENS - 1_000
    assert room([0] * 21_514, config.MAX_TOKENS) == 44_022     # the longest study task: fits a 72,000 context
    assert room([0] * 5_000, 100) == 100                       # a fork's own smaller budget stands
    with pytest.raises(ValueError):
        room([0] * config.MAX_TOKENS, config.MAX_TOKENS)
    # The key is unchanged, so rows stored before the fix keep their ids.
    c = ck.Checkpoint("anti", 1, 30, "tinker://fake/sampler_weights/000030", None, "hf/x", "vllm://anti_s1_step30")
    spec = sampling.SampleSpec(item={"task_id": "t0", "impossible_type": "conflicting"}, sample_idx=0, prompt_tokens=[1, 2, 3])
    assert sampling.spec_key(c, spec)["sampler"]["max_tokens"] == config.MAX_TOKENS


def test_rental_is_one_reservation_settled_with_the_real_bill(tmp_data):
    b = cost.Budget("M2")
    rid = rental.open_rental(b, 20.0, "vast 4xA100")
    assert b.status()["outstanding"] == 20.0
    with pytest.raises(cost.CapExceeded):
        rental.open_rental(b, 20.0, "a second machine")      # M2's cap is $25
    with pytest.raises(RuntimeError, match="reserved worst case"):
        rental.close_rental(b, rid, 21.0)                     # never settle above the reservation
    rental.close_rental(b, rid, 12.5, "invoice")
    s = b.status()
    assert s["settled"] == 12.5 and s["outstanding"] == 0
    with pytest.raises(RuntimeError, match="not an open reservation"):
        rental.close_rental(b, rid, 1.0)
    other = b.reserve(1.0, "a sample").rid
    with pytest.raises(RuntimeError, match="not a rental"):
        rental.close_rental(b, other, 0.5)


def test_register_points_a_checkpoint_at_the_server_without_changing_its_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(ck, "REBUILT_PATH", tmp_path / "rebuilt_checkpoints.json")
    before = ck.get("anti", 1, 30)
    e = vllm_client.register("anti", 1, 30, "anti_s1_step30", "abc", "vast 4xA100")
    after = ck.get("anti", 1, 30)
    assert e["published_path"] == before.sampler_path and e["backend"] == "vllm"
    assert after.model_path == "vllm://anti_s1_step30" and after.key() == before.key()
    assert vllm_client.is_vllm(after.model_path) and not vllm_client.is_vllm(ck.get("anti", 2, 60).model_path)


def _relay(handler):
    c = VllmClient("anti_s1_step30", base_url="http://x", api_key="k", transport=httpx.MockTransport(handler))
    c.relay_url, c.poll_seconds = "http://relay", 0.0
    return c


RESULT = {"token_ids": [5, 6], "logprob_tokens": [], "finish_reason": "stop", "stop_reason": None, "completion_tokens": 3}


def test_relay_path_submits_under_the_given_id_and_survives_dropped_connections():
    calls = {"post": 0, "state": 0, "get": 0}

    def handler(request):
        path = request.url.path
        if request.method == "POST":
            calls["post"] += 1
            if calls["post"] == 1:
                raise httpx.ConnectError("tunnel down")          # the first submit never arrives
            req = json.loads(request.content)
            assert req["id"] == "abc123" and req["body"]["prompt"] == [1, 2, 3]
            return httpx.Response(200, json={"status": "running"})
        if path == "/state":
            calls["state"] += 1
            if calls["state"] == 1:
                raise httpx.ReadError("tunnel dropped")
            done = calls["state"] >= 4
            return httpx.Response(200, json={"done": ["abc123"] if done else [], "active": [] if done else ["abc123"]})
        calls["get"] += 1
        return httpx.Response(200, json={"status": "done", "result": RESULT})

    r = asyncio.run(_relay(handler).sample([1, 2, 3], 100, 1.0, [STOP], job_id="abc123"))
    assert r.tokens == [5, 6, STOP] and r.stop_reason == "stop"
    assert calls["post"] == 2 and calls["get"] == 1


@pytest.mark.parametrize("answer,why", [
    ({"status": "unknown"}, "lost the job"),
    ({"status": "error", "error": "vLLM answered 400: too long"}, "400"),
])
def test_relay_reports_a_failed_or_lost_job(answer, why):
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"status": "running"})
        if request.url.path == "/state":
            return httpx.Response(200, json={"done": [], "active": []})
        return httpx.Response(200, json=answer)

    with pytest.raises(VllmError, match=why):
        run(_relay(handler))
