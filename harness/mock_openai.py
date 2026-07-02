"""A real-socket mock of an OpenAI-compatible streaming chat endpoint.

Streams SSE chunks over a genuine localhost TCP socket (with ``TCP_NODELAY``) at *injected*
delays, so the harness's TTFT / ITL / throughput math can be verified deterministically with
real network framing — no model, GPU or external service needed. An in-process ASGI mock can't
be used here: httpx's ASGITransport buffers the whole response, collapsing inter-token gaps to
near-zero and hiding streaming bugs.

Run standalone as a no-GPU demo endpoint:
    python mock_openai.py --port 8000 --ttft-ms 120 --itl-ms 25 --tokens 256
then point a harness config's ``engine.base_url`` at it.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import socket
from typing import AsyncIterator


def _chunk(payload: str) -> bytes:
    """Wraps an SSE ``data:`` line as one HTTP/1.1 chunked-transfer chunk."""
    body = payload.encode()
    return f"{len(body):X}\r\n".encode() + body + b"\r\n"


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, *,
                  ttft_ms: float, itl_ms: float, n_tokens: int, include_usage: bool, fail: bool) -> None:
    """Serves one request: an immediate role-only delta, then ``n_tokens`` content chunks."""
    with contextlib.suppress(Exception):
        await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)  # consume request headers

    sock = writer.get_extra_info("socket")
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    if fail:
        body = b'{"error":"boom"}'
        writer.write(b"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
        with contextlib.suppress(Exception):
            await writer.drain()
            writer.close()
        return

    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                 b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
    await writer.drain()

    base = {"id": "mock", "object": "chat.completion.chunk"}

    async def emit(obj: dict) -> None:
        writer.write(_chunk(f"data: {json.dumps(obj)}\n\n"))
        await writer.drain()

    await emit({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
    for i in range(n_tokens):
        await asyncio.sleep((ttft_ms if i == 0 else itl_ms) / 1000.0)
        await emit({**base, "choices": [{"index": 0, "delta": {"content": "tok "}}]})
    if include_usage:
        await emit({**base, "choices": [], "usage": {"completion_tokens": n_tokens, "prompt_tokens": 10}})

    writer.write(_chunk("data: [DONE]\n\n"))
    writer.write(b"0\r\n\r\n")  # terminating chunk
    with contextlib.suppress(Exception):
        await writer.drain()
        writer.close()


@contextlib.asynccontextmanager
async def serve_mock(ttft_ms: float = 100.0, itl_ms: float = 20.0, n_tokens: int = 8,
                     include_usage: bool = True, fail: bool = False) -> AsyncIterator[str]:
    """Runs the mock on an ephemeral localhost port for the duration of the context.

    Yields the base URL (e.g. ``http://127.0.0.1:54321``) to point an AsyncClient at.
    """
    async def cb(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        await _handle(r, w, ttft_ms=ttft_ms, itl_ms=itl_ms, n_tokens=n_tokens,
                      include_usage=include_usage, fail=fail)

    server = await asyncio.start_server(cb, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server:
        yield f"http://127.0.0.1:{port}"


def _main() -> None:
    """Runs the mock as a long-lived demo endpoint."""
    ap = argparse.ArgumentParser(description="No-GPU mock OpenAI streaming endpoint")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ttft-ms", type=float, default=120.0)
    ap.add_argument("--itl-ms", type=float, default=25.0)
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    async def run() -> None:
        async def cb(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
            await _handle(r, w, ttft_ms=args.ttft_ms, itl_ms=args.itl_ms, n_tokens=args.tokens,
                          include_usage=True, fail=False)
        server = await asyncio.start_server(cb, "127.0.0.1", args.port)
        print(f"mock OpenAI endpoint: http://127.0.0.1:{args.port}  "
              f"(ttft={args.ttft_ms}ms itl={args.itl_ms}ms tokens={args.tokens})")
        async with server:
            await server.serve_forever()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())


if __name__ == "__main__":
    _main()
