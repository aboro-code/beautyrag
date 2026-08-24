"""Benchmark: real /ask latency, Redis cache miss vs. cache hit.

Cache mechanics (confirmed against app/cache.py and app/routers/ask.py): the cache
key is `f"ask:{normalize_query(question)}"` (lowercased, whitespace-collapsed
question text), TTL is `settings.ask_cache_ttl_seconds` (300s). A cache hit returns
straight from Redis; a miss runs the full RAG pipeline (hybrid search + review
fetch + a real Groq LLM call) and then writes the result to Redis before returning.
`/ask` is also rate-limited (30 req/min/IP by default, app/rate_limit.py) --
enforced on every request regardless of hit/miss, so this script runs its own local
server with that limit raised via env var, to avoid the benchmark contaminating its
own timings with 429s.

Methodology note / deviation from a literal "clear cache once, then run pairs of
miss/hit batches" reading: if the cache is only cleared once at the very start, only
the *first* repetition's "miss" batch is a genuine miss -- every later "miss" batch
would silently already be cached from the prior hit batch (TTL is 300s, comfortably
longer than a full repetition takes), making its timings actually cache-hit numbers
mislabeled as misses. To get N genuine miss samples and N genuine hit samples, the
cache is cleared before every repetition's miss batch, not just once overall.

Reuses all 20 of the project's existing curated query strings from
eval_ground_truth_labels.json (12 labeled + 8 excluded -- the "excluded" ones are
still perfectly fine natural-language questions, they just lack SQL-derivable
retrieval ground truth, which is irrelevant for a pure latency benchmark) rather
than inventing a new list.

Run: python -m eval.latency_bench
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from redis.asyncio import from_url

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import settings  # noqa: E402

REPETITIONS = 3
# Groq's free-tier rate limit (observed: a 429 after ~7-8 back-to-back /ask calls,
# each of which makes one real Groq request) is stricter than our own app's 30/min
# limiter. This paces miss-pass requests only (hit-pass never calls Groq) to stay
# under it -- a benchmark-script concern, not an app behavior change.
MISS_REQUEST_DELAY_S = 3.0
BENCH_PORT = 8001
BENCH_BASE_URL = f"http://127.0.0.1:{BENCH_PORT}"
BENCH_RATE_LIMIT_OVERRIDE = "100000"  # effectively unlimited for this local run only

GROUND_TRUTH_PATH = Path(__file__).parent / "eval_ground_truth_labels.json"


def load_queries() -> list[str]:
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    queries = [q["query_text"] for q in gt["labeled_queries"].values()]
    queries += [q["query_text"] for q in gt["excluded_queries"].values()]
    return queries


async def clear_ask_cache() -> int:
    redis_client = from_url(settings.redis_url, decode_responses=True)
    keys = [k async for k in redis_client.scan_iter(match="ask:*")]
    if keys:
        await redis_client.delete(*keys)
    await redis_client.aclose()
    return len(keys)


def percentiles(samples_ms: list[float]) -> tuple[float, float]:
    arr = np.array(samples_ms)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


async def time_request(client: httpx.AsyncClient, question: str) -> float:
    for attempt in (1, 2):
        start = time.perf_counter()
        try:
            resp = await client.post(f"{BENCH_BASE_URL}/ask", json={"question": question}, timeout=60)
            resp.raise_for_status()
            return (time.perf_counter() - start) * 1000
        except httpx.HTTPError as e:
            if attempt == 2:
                raise
            # Our app doesn't propagate Groq's actual status code -- an unhandled
            # Groq 429 just surfaces as our own opaque 500 (confirmed via the
            # server log during development of this script). Treat any 500 as
            # possibly that and back off longer; anything else gets a short retry.
            likely_upstream_429 = isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 500
            backoff_s = 20.0 if likely_upstream_429 else 2.0
            print(f"    [retry] request failed ({e}), waiting {backoff_s:.0f}s and retrying once...")
            await asyncio.sleep(backoff_s)


async def run_benchmark(queries: list[str]) -> tuple[list[float], list[float]]:
    miss_samples: list[float] = []
    hit_samples: list[float] = []

    async with httpx.AsyncClient() as client:
        for rep in range(1, REPETITIONS + 1):
            cleared = await clear_ask_cache()
            print(f"Repetition {rep}/{REPETITIONS}: cleared {cleared} cached ask: keys")

            print(f"  Miss pass ({len(queries)} queries)...")
            for i, q in enumerate(queries):
                if i > 0:
                    await asyncio.sleep(MISS_REQUEST_DELAY_S)
                ms = await time_request(client, q)
                miss_samples.append(ms)
                print(f"    {ms:9.2f}ms  {q[:60]}")

            print(f"  Hit pass ({len(queries)} queries)...")
            for q in queries:
                ms = await time_request(client, q)
                hit_samples.append(ms)
                print(f"    {ms:9.2f}ms  {q[:60]}")

    return miss_samples, hit_samples


async def wait_for_health(client: httpx.AsyncClient, timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            resp = await client.get(f"{BENCH_BASE_URL}/health", timeout=2)
            if resp.status_code == 200:
                return
        except httpx.RequestError:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError("Benchmark server did not become healthy in time")


async def main() -> None:
    queries = load_queries()
    print(f"Loaded {len(queries)} queries.")

    env = os.environ.copy()
    env["RATE_LIMIT_PER_MINUTE"] = BENCH_RATE_LIMIT_OVERRIDE

    log_path = Path(__file__).parent / "_latency_bench_server.log"
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(BENCH_PORT)],
        cwd=str(Path(__file__).parent.parent),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        async with httpx.AsyncClient() as client:
            await wait_for_health(client)
        print("Benchmark server is up.\n")

        miss_samples, hit_samples = await run_benchmark(queries)
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    miss_p50, miss_p95 = percentiles(miss_samples)
    hit_p50, hit_p95 = percentiles(hit_samples)

    print("\n" + "=" * 60)
    print(f"Cache MISS -- n={len(miss_samples)}")
    print(f"  p50: {miss_p50:.2f}ms")
    print(f"  p95: {miss_p95:.2f}ms")
    print(f"  min: {min(miss_samples):.2f}ms  max: {max(miss_samples):.2f}ms")
    print(f"\nCache HIT -- n={len(hit_samples)}")
    print(f"  p50: {hit_p50:.2f}ms")
    print(f"  p95: {hit_p95:.2f}ms")
    print(f"  min: {min(hit_samples):.2f}ms  max: {max(hit_samples):.2f}ms")
    if len(miss_samples) < 100 or len(hit_samples) < 100:
        print(
            f"\nSample size warning: n={len(miss_samples)} per group is small for a "
            f"p95 estimate -- with {REPETITIONS} repetitions x {len(queries)} "
            f"queries, p95 is roughly the {int(len(miss_samples) * 0.95)}th of "
            f"{len(miss_samples)} sorted samples, so a single slow outlier (e.g. one "
            f"unusually slow Groq response) can move it noticeably. Treat as "
            f"directionally accurate, not a tight bound."
        )


if __name__ == "__main__":
    asyncio.run(main())
