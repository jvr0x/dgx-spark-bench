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

from bench import EngineTarget, SweepConfig, Workload, nvidia_smi_stats, percentile, run_point, stream_request
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


@pytest.mark.asyncio
async def test_run_point_includes_gpu_snapshot_when_available():
    """run_point attaches a ``gpu`` object with temp/power after each concurrency point."""
    from unittest.mock import patch  # noqa: PLC0415
    fake_smi = {"gpu_index": 0, "temperature_c": 72, "power_draw_w": 185.32, "power_limit_w": 400}
    async with serve_mock(ttft_ms=20, itl_ms=5, n_tokens=4) as url:
        sweep = SweepConfig(concurrencies=[1], warmup_s=0.1, measure_s=0.5)
        with patch("bench.nvidia_smi_stats", return_value=fake_smi):
            pt = await run_point(EngineTarget(base_url=url, model="m"), WL, 1, sweep)
    assert "gpu" in pt
    assert pt["gpu"]["temperature_c"] == 72
    assert pt["gpu"]["power_draw_w"] == pytest.approx(185.32)


@pytest.mark.asyncio
async def test_run_point_omits_gpu_snapshot_when_unavailable():
    """run_point does not include ``gpu`` when nvidia-smi returns None."""
    from unittest.mock import patch  # noqa: PLC0415
    async with serve_mock(ttft_ms=20, itl_ms=5, n_tokens=4) as url:
        sweep = SweepConfig(concurrencies=[1], warmup_s=0.1, measure_s=0.5)
        with patch("bench.nvidia_smi_stats", return_value=None):
            pt = await run_point(EngineTarget(base_url=url, model="m"), WL, 1, sweep)
    assert "gpu" not in pt


# ── nvidia-smi parsing tests ──────────────────────────────────────────────────


class TestNvidiaSmiParse:
    """Unit tests for ``nvidia_smi_stats()`` CSV parsing logic.

    We test the parser against canned CSV strings (the exact format ``nvidia-smi
    --query-gpu=index,temperature.gpu,power.draw,power.limit --format=csv`` emits)
    rather than shelling out, so every test is deterministic and GPU-free.
    """

    def _parse(self, csv_text: str) -> dict[str, object] | None:
        """Helper: feed *csv_text* through the same parser the harness uses."""
        import subprocess  # noqa: PLC0415 – inside test to avoid import overhead
        # We can't mock ``subprocess.run`` easily from here, so instead we test the
        # private parsing logic directly by patching the call.  For clarity we simply
        # exercise the real code path with ``side_effect`` on ``subprocess.run``.
        from unittest.mock import patch  # noqa: PLC0415
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = csv_text
            result = nvidia_smi_stats()
        return result

    def test_normal_values(self):
        """Typical output: integer temperature, float power.draw, integer power.limit."""
        # --format=csv,noheader => no header line
        csv_text = "0, 72, 185.32 W, 400 W\n"
        result = self._parse(csv_text)
        assert result is not None
        assert result["gpu_index"] == 0
        assert result["temperature_c"] == 72
        assert result["power_draw_w"] == pytest.approx(185.32)
        assert result["power_limit_w"] == 400

    def test_power_limit_na(self):
        """Some GPUs report [N/A] for power limit — should be ``None``."""
        csv_text = "0, 61, 39.04 W, [N/A]\n"
        result = self._parse(csv_text)
        assert result is not None
        assert result["temperature_c"] == 61
        assert result["power_draw_w"] == pytest.approx(39.04)
        assert result["power_limit_w"] is None

    def test_temperature_na(self):
        """[N/A] temperature should yield ``None``."""
        csv_text = "0, [N/A], 50.0 W, 100 W\n"
        result = self._parse(csv_text)
        assert result is not None
        assert result["temperature_c"] is None
        assert result["power_draw_w"] == pytest.approx(50.0)

    def test_multiple_gpus_takes_first(self):
        """Multi-GPU output: parser returns the first row (index 0)."""
        csv_text = "0, 65, 120.0 W, 400 W\n1, 68, 125.5 W, 400 W\n"
        result = self._parse(csv_text)
        assert result is not None
        assert result["gpu_index"] == 0
        assert result["temperature_c"] == 65
        assert result["power_draw_w"] == pytest.approx(120.0)

    def test_empty_output(self):
        """Empty stdout returns ``None``."""
        result = self._parse("")
        assert result is None

    def test_no_data_rows(self):
        """Whitespace-only output returns ``None``."""
        result = self._parse("\n")
        assert result is None

    def test_parse_command_failure(self):
        """When nvidia-smi fails (non-zero exit), returns ``None``."""
        from unittest.mock import patch, MagicMock  # noqa: PLC0415
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = nvidia_smi_stats()
        assert result is None
