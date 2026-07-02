"""Deterministic tests for the harness measurement math.

Each test points the harness at a real-socket mock endpoint with *injected* delays, so
TTFT / ITL / token-count / aggregation logic is verified exactly against genuine streaming
framing — no GPU, model or external service. Covers the expected path, two edge cases and a
failure case, plus the percentile helper.
"""
from __future__ import annotations

import statistics

import httpx
import pytest

from bench import EngineTarget, SweepConfig, Workload, percentile, run_point, stream_request
from mock_openai import serve_mock

WL = Workload(name="chat", prompt_tokens=16, output_tokens=8)


async def _one(url: str) -> "RequestResult":  # noqa: F821 - return type documented in bench
    """Issues a single streaming request against ``url`` and returns its result."""
    async with httpx.AsyncClient(base_url=url) as c:
        return await stream_request(c, EngineTarget(base_url=url, model="m"), WL)


@pytest.mark.asyncio
async def test_expected_metrics():
    """Happy path: TTFT clocked at first content (role delta ignored), ITLs and usage correct."""
    async with serve_mock(ttft_ms=100, itl_ms=20, n_tokens=6) as url:
        r = await _one(url)
    assert r.ok
    assert r.decode_tokens == 6                       # read from the usage chunk
    assert 60 <= r.ttft_ms <= 250                     # ~100ms; the role-only delta is not counted
    assert len(r.itls_ms) == 5                        # 6 tokens -> 5 inter-token gaps
    assert 8 <= statistics.mean(r.itls_ms) <= 60      # ~20ms each, over a real socket


@pytest.mark.asyncio
async def test_edge_no_usage_falls_back_to_chunk_count():
    """When the server omits usage, decode_tokens falls back to counted content chunks."""
    async with serve_mock(n_tokens=4, include_usage=False) as url:
        r = await _one(url)
    assert r.ok and r.decode_tokens == 4


@pytest.mark.asyncio
async def test_edge_single_token_has_no_itls():
    """One token => a TTFT but zero inter-token gaps; decode_rate must not divide by zero."""
    async with serve_mock(ttft_ms=30, n_tokens=1) as url:
        r = await _one(url)
    assert r.ok and r.decode_tokens == 1 and r.itls_ms == []
    assert r.decode_rate >= 0.0


@pytest.mark.asyncio
async def test_failure_http_500_marks_request_failed():
    """A server error fails the single request, not the whole run."""
    async with serve_mock(fail=True) as url:
        r = await _one(url)
    assert not r.ok and r.error


def test_percentile_math():
    """Linear-interpolated percentile, with empty-list and single-element guards."""
    assert percentile([], 50) == 0.0
    assert percentile([10], 99) == 10
    assert percentile([1, 2, 3, 4], 50) == 2.5


@pytest.mark.asyncio
async def test_run_point_aggregates_window():
    """A 2-session point over a short window produces sane, ordered metrics."""
    async with serve_mock(ttft_ms=20, itl_ms=5, n_tokens=4) as url:
        sweep = SweepConfig(concurrencies=[2], warmup_s=0.2, measure_s=1.0)
        pt = await run_point(EngineTarget(base_url=url, model="m"), WL, 2, sweep)
    assert pt["concurrency"] == 2
    assert pt["samples"] >= 1
    assert pt["agg_tok_s"] > 0
    assert pt["ttft_ms"]["p95"] >= pt["ttft_ms"]["p50"]
