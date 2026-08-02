"""Decode-rate vs draft-acceptance probe for the 0731 dual profile.

Shows why one number does not describe this server: engine step rate is
constant, so decode throughput is set almost entirely by how many speculative
tokens get accepted per step, which is a property of the text being generated.

All rates are taken from vLLM's own metrics (``generation_tokens_total`` over
``request_decode_time_seconds``), so the Python client is not in the
measurement path. Run it against a live server:

    python probe-decode-shapes.py            # defaults to localhost:8888
"""
from __future__ import annotations

import json
import time

import httpx

BASE = "http://localhost:8888"
MODEL = "deepseek-v4-flash-0731-dual"
KEYS = (
    "vllm:generation_tokens_total",
    "vllm:request_decode_time_seconds_sum",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)

FILLER = "Summarise the following text.\n\n" + "benchmark " * 192
LIST_INSTR = "\nReturn exactly 128 numbered lowercase English words, then stop."
PROSE = "Explain how tensor parallelism splits a transformer layer across two GPUs, then give a short worked example."


def snapshot() -> dict[str, float]:
    """Returns the counters this probe differences, as a name -> value map."""
    out = dict.fromkeys(KEYS, 0.0)
    for line in httpx.get(f"{BASE}/metrics", timeout=30).text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in out:
            out[name] = float(line.rsplit(" ", 1)[1])
    return out


def run(label: str, prompt: str, **params: object) -> None:
    """Streams one request and prints server-measured decode rate and acceptance."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        **params,
    }
    before = snapshot()
    first = None
    usage = None
    with httpx.stream("POST", f"{BASE}/v1/chat/completions", json=body, timeout=900) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            event = json.loads(line[6:])
            if first is None and any(
                c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content")
                for c in event.get("choices", [])
            ):
                first = time.perf_counter()
            if event.get("usage"):
                usage = event["usage"]
    after = snapshot()

    d = {k: after[k] - before[k] for k in KEYS}
    gen = d["vllm:generation_tokens_total"]
    dec_s = d["vllm:request_decode_time_seconds_sum"]
    drafts = d["vllm:spec_decode_num_drafts_total"]
    draft_toks = d["vllm:spec_decode_num_draft_tokens_total"]
    accepted = d["vllm:spec_decode_num_accepted_tokens_total"]
    # Reason: a step emits the accepted draft tokens plus the always-emitted
    # bonus token, so tokens/step is (accepted + drafts) / drafts.
    per_step = (accepted + drafts) / drafts if drafts else float("nan")
    accept = accepted / draft_toks if draft_toks else float("nan")
    rate = gen / dec_s if dec_s else float("nan")
    print(
        f"{label:<44} out={usage['completion_tokens']:>4}  decode={rate:6.1f} tok/s  "
        f"accept={accept:5.1%}  tok/step={per_step:4.2f}  steps/s={rate / per_step:5.1f}"
    )


if __name__ == "__main__":
    run("warmup", PROSE, max_tokens=64, temperature=0.0)

    print("\n-- highly draftable output (upstream benchmark-0731.py's shape) --")
    run("numbered word list, temp 0.6 natural", FILLER + LIST_INSTR, max_tokens=600, temperature=0.6, top_p=0.95)
    run("numbered word list, temp 1.0 forced 512", FILLER + LIST_INSTR, max_tokens=512, min_tokens=512, ignore_eos=True, temperature=1.0)

    print("\n-- free-form prose --")
    run("prose answer, temp 0.0 natural", PROSE, max_tokens=512, temperature=0.0)
    run("prose answer, temp 1.0 natural", PROSE, max_tokens=512, temperature=1.0)

    print("\n-- this repo's harness shape (filler prompt, forced past EOS) --")
    run("filler summarise, forced 256", FILLER, max_tokens=256, min_tokens=256, ignore_eos=True, temperature=1.0)
    run("filler summarise, forced 512", FILLER, max_tokens=512, min_tokens=512, ignore_eos=True, temperature=1.0)
