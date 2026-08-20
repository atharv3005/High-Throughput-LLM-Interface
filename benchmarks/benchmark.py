import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


URL = "http://localhost:8080/generate"

PROMPT = (
    "Explain what PagedAttention is in vLLM. "
    "Focus on KV-cache management and memory efficiency."
)


def send_request(request_id: int, max_tokens: int) -> dict:
    start = time.perf_counter()

    response = requests.post(
        URL,
        json={
            "prompt": PROMPT,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )

    elapsed = time.perf_counter() - start
    response.raise_for_status()

    data = response.json()

    return {
        "request_id": request_id,
        "latency_seconds": elapsed,
        "completion_tokens": data["completion_tokens"],
        "total_tokens": data["total_tokens"],
    }


def warmup():
    print("Warmup...")
    result = send_request(-1, 32)
    print(
        f"Warmup complete: {result['completion_tokens']} tokens\n"
    )


def run_level(
    concurrency: int,
    requests_count: int,
    max_tokens: int,
) -> dict:

    print(
        f"Running concurrency={concurrency}, "
        f"requests={requests_count}, "
        f"max_tokens={max_tokens}"
    )

    start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                send_request,
                i,
                max_tokens,
            )
            for i in range(requests_count)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    wall_time = time.perf_counter() - start

    latencies = [
        result["latency_seconds"]
        for result in results
    ]

    total_completion_tokens = sum(
        result["completion_tokens"]
        for result in results
    )

    sorted_latencies = sorted(latencies)

    p95_index = min(
        len(sorted_latencies) - 1,
        int(0.95 * len(sorted_latencies)),
    )

    result = {
        "concurrency": concurrency,
        "requests": requests_count,
        "max_tokens": max_tokens,
        "wall_time_seconds": round(
            wall_time,
            4,
        ),
        "total_completion_tokens": (
            total_completion_tokens
        ),
        "aggregate_throughput_tok_s": round(
            total_completion_tokens / wall_time,
            2,
        ),
        "requests_per_second": round(
            requests_count / wall_time,
            2,
        ),
        "mean_latency_seconds": round(
            statistics.mean(latencies),
            4,
        ),
        "p50_latency_seconds": round(
            statistics.median(latencies),
            4,
        ),
        "p95_latency_seconds": round(
            sorted_latencies[p95_index],
            4,
        ),
        "max_latency_seconds": round(
            max(latencies),
            4,
        ),
        "requests_detail": sorted(
            results,
            key=lambda item: item["request_id"],
        ),
    }

    print(
        f"  throughput: "
        f"{result['aggregate_throughput_tok_s']} tok/s"
    )
    print(
        f"  mean latency: "
        f"{result['mean_latency_seconds']} s"
    )
    print(
        f"  p50 latency: "
        f"{result['p50_latency_seconds']} s"
    )
    print(
        f"  p95 latency: "
        f"{result['p95_latency_seconds']} s\n"
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 2, 4],
    )

    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--output",
        default="results/m4-gateway-benchmark.json",
    )

    args = parser.parse_args()

    warmup()

    all_results = []

    for concurrency in args.concurrency:

        all_results.append(
            run_level(
                concurrency,
                args.requests_per_level,
                args.max_tokens,
            )
        )

    output = {
        "benchmark": "M4",
        "architecture": (
            "FastAPI gateway -> native vLLM server"
        ),
        "endpoint": URL,
        "model": (
            "meta-llama/"
            "Llama-3.2-1B-Instruct"
        ),
        "prompt": PROMPT,
        "results": all_results,
    }

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"Results written to {args.output}"
    )


if __name__ == "__main__":
    main()
