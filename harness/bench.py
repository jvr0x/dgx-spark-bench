"""DGX Spark concurrency-throughput benchmark harness.

Drives a *closed-loop* concurrency sweep against any OpenAI-compatible streaming
endpoint (vLLM, SGLang, llama.cpp, Ollama), measures time-to-first-token, inter-token
latency and throughput, and emits a ``results.json`` that conforms to
``../schema/results.schema.json``.

Design notes (the parts that make the numbers trustworthy):

* **Forced output length** — ``ignore_eos`` / ``min_tokens`` (passed via ``engine.extra_body``)
  make every request emit exactly ``output_tokens``; we still count the *actual*
  ``completion_tokens`` from the usage chunk and never assume the cap.
* **Window aggregate** — aggregate throughput is ``tokens completed in the steady-state
  window / window wall-seconds``, not ``mean(per-request rate) x N``.
* **Per-session rate** — mean of each request's own decode rate; reported separately.
* **Client pool >= N** — one ``AsyncClient`` per concurrency point sized to N so the client
  never serialises and we measure the server, not the harness.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

HARNESS_VERSION = "0.1.0"
SCHEMA_VERSION = "1.0"


@dataclass
class Workload:
    """A fixed request shape used identically across every configuration."""

    name: str
    prompt_tokens: int
    output_tokens: int

    def prompt(self, nonce: str = "") -> str:
        """Returns a deterministic prompt sized to roughly ``prompt_tokens`` tokens.

        ``nonce`` is prepended so each request has a unique prefix. This defeats
        server-side prefix caching (e.g. vLLM ``--enable-prefix-caching``), which would
        otherwise let identical prompts skip prefill and report unrealistically low TTFT
        and inflated throughput. The body is otherwise identical across configs, so
        prefill cost stays comparable. ~0.75 tokens/word is assumed for sizing only.
        """
        lead = f"[bench {nonce}] " if nonce else ""
        word = "benchmark "
        n_words = max(1, int(self.prompt_tokens * 0.75))
        return (lead + "Summarise the following text.\n\n" + word * n_words).strip()


@dataclass
class EngineTarget:
    """Where to send requests and the served model id."""

    base_url: str
    model: str
    api_key: str = "dummy"
    # Backend-specific knobs, e.g. {"ignore_eos": True, "min_tokens": 256} for vLLM/SGLang.
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepConfig:
    """Concurrency levels and the timing of each measurement point."""

    concurrencies: list[int]
    warmup_s: float = 8.0
    measure_s: float = 30.0
    request_timeout_s: float = 300.0


@dataclass
class RequestResult:
    """Timings and token count for a single completed (or failed) request."""

    ok: bool
    start: float
    first_token: float | None
    end: float
    decode_tokens: int
    itls_ms: list[float]
    error: str | None = None

    @property
    def ttft_ms(self) -> float:
        """Time from request send to the first non-empty content token, in ms."""
        return (self.first_token - self.start) * 1000.0 if self.first_token else float("nan")

    @property
    def decode_rate(self) -> float:
        """This request's own decode throughput (tokens / decode-seconds)."""
        if not self.first_token:
            return 0.0
        dur = self.end - self.first_token
        return self.decode_tokens / dur if dur > 0 else 0.0


def percentile(xs: list[float], p: float) -> float:
    """Returns the linear-interpolated p-th percentile of ``xs`` (0 if empty)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def stream_request(client: httpx.AsyncClient, target: EngineTarget, workload: Workload,
                         nonce: str = "") -> RequestResult:
    """Issues one streaming chat completion and records its timings and token count.

    TTFT is clocked at the first chunk carrying a non-empty text delta — ``content``,
    ``reasoning_content``, or ``reasoning`` (reasoning parsers route thinking tokens to the
    latter fields; role-only deltas are ignored). ``completion_tokens`` is read from the
    final usage chunk when present, falling back to the streamed text-chunk count.
    ``nonce`` makes the prompt unique per request to defeat prefix caching.
    """
    body: dict[str, Any] = {
        "model": target.model,
        "messages": [{"role": "user", "content": workload.prompt(nonce)}],
        "max_tokens": workload.output_tokens,
        "temperature": 0,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        **target.extra_body,
    }
    start = time.monotonic()
    first: float | None = None
    last_t: float | None = None
    itls_ms: list[float] = []
    content_chunks = 0
    usage_tokens: int | None = None
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    usage_tokens = int(usage["completion_tokens"])
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                # Reasoning parsers (e.g. vLLM's qwen3) route thinking tokens to a separate
                # delta field; they are decode tokens all the same. Field name varies by build.
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                if piece:
                    now = time.monotonic()
                    if first is None:
                        first = now
                    elif last_t is not None:
                        itls_ms.append((now - last_t) * 1000.0)
                    last_t = now
                    content_chunks += 1
        end = time.monotonic()
        decode_tokens = usage_tokens if usage_tokens is not None else content_chunks
        return RequestResult(True, start, first, end, decode_tokens, itls_ms)
    except Exception as exc:  # noqa: BLE001 - any transport/parse error fails the request, not the run
        return RequestResult(False, start, first, time.monotonic(), 0, itls_ms, error=str(exc))


async def _worker(client: httpx.AsyncClient, target: EngineTarget, workload: Workload,
                  stop_at: float, sink: list[RequestResult], wid: int) -> None:
    """Closed-loop worker: issues back-to-back requests until ``stop_at`` (monotonic).

    Each request gets a unique ``wid-i`` nonce so no two prompts share a prefix.
    """
    i = 0
    while time.monotonic() < stop_at:
        sink.append(await stream_request(client, target, workload, nonce=f"{wid}-{i}"))
        i += 1


def _client_for(target: EngineTarget, n: int, sweep: SweepConfig) -> httpx.AsyncClient:
    """Builds an AsyncClient whose connection pool is sized to N (so it never serialises)."""
    limits = httpx.Limits(max_connections=n + 4, max_keepalive_connections=n + 4)
    return httpx.AsyncClient(
        base_url=target.base_url,
        limits=limits,
        timeout=httpx.Timeout(sweep.request_timeout_s),
        headers={"Authorization": f"Bearer {target.api_key}"},
    )


async def run_point(target: EngineTarget, workload: Workload, n: int, sweep: SweepConfig) -> dict[str, Any]:
    """Runs one concurrency level and returns a schema ``points[]`` entry.

    Only requests that *complete inside the steady-state window* count toward the metrics;
    warmup and drain are excluded.
    """
    results: list[RequestResult] = []
    async with _client_for(target, n, sweep) as client:
        t0 = time.monotonic()
        window_start = t0 + sweep.warmup_s
        window_end = window_start + sweep.measure_s
        await asyncio.gather(*[
            asyncio.create_task(_worker(client, target, workload, window_end, results, wid))
            for wid in range(n)
        ])

    in_win = [r for r in results if r.ok and r.first_token and window_start <= r.end <= window_end]
    total_tokens = sum(r.decode_tokens for r in in_win)
    ttfts = [r.ttft_ms for r in in_win]
    rates = [r.decode_rate for r in in_win]
    all_itls = [x for r in in_win for x in r.itls_ms]
    failures = sum(1 for r in results if not r.ok)

    return {
        "concurrency": n,
        "agg_tok_s": round(total_tokens / sweep.measure_s, 1),
        "per_session_tok_s": round(statistics.mean(rates), 1) if rates else 0.0,
        "ttft_ms": {"p50": round(percentile(ttfts, 50)), "p95": round(percentile(ttfts, 95)),
                    "p99": round(percentile(ttfts, 99))},
        "itl_ms": {"p50": round(percentile(all_itls, 50), 1), "p95": round(percentile(all_itls, 95), 1),
                   "p99": round(percentile(all_itls, 99), 1)},
        "samples": len(in_win),
        **({"_failures": failures} if failures else {}),
    }


def _git_commit() -> str:
    """Returns the short git commit of the harness, or 'unknown' if not a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


async def run_sweep(cfg: dict[str, Any], timestamp: str) -> dict[str, Any]:
    """Runs the full sweep described by ``cfg`` and returns a complete results document."""
    workload = Workload(**cfg["workload"])
    target = EngineTarget(**cfg["engine"])
    sweep = SweepConfig(**cfg["sweep"])
    series_meta = cfg["series"]

    points: list[dict[str, Any]] = []
    for n in sweep.concurrencies:
        print(f"  N={n:>4} ...", end="", flush=True)
        point = await run_point(target, workload, n, sweep)
        points.append(point)
        print(f" agg={point['agg_tok_s']:>7} tok/s  per={point['per_session_tok_s']:>6}  "
              f"ttft_p95={point['ttft_ms']['p95']:>5} ms  (n={point['samples']})")

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "id": cfg.get("run_id", f"{series_meta['id']}-{timestamp}"),
            "timestamp": timestamp,
            "harness_version": HARNESS_VERSION,
            "git_commit": _git_commit(),
            "hardware": cfg["hardware"],
            "workload": {**cfg["workload"], "mode": "closed_loop"},
            "sweep": cfg["sweep"],
        },
        "series": [{**series_meta, "points": points}],
    }


def _validate(doc: dict[str, Any]) -> None:
    """Validates ``doc`` against the JSON schema if ``jsonschema`` is installed."""
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "results.schema.json"
    try:
        import jsonschema  # noqa: PLC0415 - optional dependency
    except ImportError:
        print("  (jsonschema not installed — skipping validation)")
        return
    jsonschema.validate(doc, json.loads(schema_path.read_text()))
    print("  schema: OK")


def main() -> None:
    """CLI entry point: loads a config, runs the sweep, writes and validates results.json."""
    ap = argparse.ArgumentParser(description="DGX Spark concurrency-throughput benchmark")
    ap.add_argument("config", type=Path, help="YAML run config (see configs/)")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output results.json path")
    ap.add_argument("--timestamp", default=None, help="ISO-8601 run timestamp (default: now)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    timestamp = args.timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print(f"running sweep: {cfg['series']['id']}  ->  {args.out}")
    doc = asyncio.run(run_sweep(cfg, timestamp))
    _validate(doc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
