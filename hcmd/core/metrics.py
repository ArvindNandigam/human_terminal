"""Performance measurement helpers for hcmd."""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Iterable


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, round((percentile_value / 100) * (len(ordered) - 1))),
    )
    return ordered[index]


def benchmark(
    operation: Callable[[str], Any],
    commands: Iterable[str],
) -> dict[str, Any]:
    samples = list(commands)
    durations_ms: list[float] = []
    successes = 0

    for command in samples:
        started = time.perf_counter()
        result = operation(command)
        durations_ms.append((time.perf_counter() - started) * 1000)
        if result:
            successes += 1

    total_seconds = sum(durations_ms) / 1000
    return {
        "samples": len(samples),
        "successful_samples": successes,
        "success_rate": successes / len(samples) if samples else 0.0,
        "average_ms": statistics.mean(durations_ms) if durations_ms else 0.0,
        "median_ms": statistics.median(durations_ms) if durations_ms else 0.0,
        "p95_ms": percentile(durations_ms, 95),
        "throughput_per_second": len(samples) / total_seconds if total_seconds else 0.0,
    }
