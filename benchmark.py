"""Simple benchmark script for OptiPrompt throughput and latency."""

import statistics
import time

from app.core.pipeline import OptiPromptPipeline, PipelineConfig

SAMPLE_PROMPTS = [
    "Please make sure to create a detailed API design document that explains all endpoints and includes examples for each one in order to help the team implement it.",
    "I would like you to basically analyze this report and provide a summary that is very clear and also includes actionable next steps for improving response times.",
    "The system should be able to generate an optimization plan due to the fact that token costs are too high and we need a compact, structured prompt format.",
]


def run_benchmark(iterations: int = 50) -> None:
    pipeline = OptiPromptPipeline()
    cfg = PipelineConfig(mode="balanced", seed=42, include_candidates=False, debug=False)

    latencies = []
    for i in range(iterations):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        start = time.perf_counter()
        pipeline.optimize(prompt, cfg)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    print("Benchmark Results")
    print(f"Iterations: {iterations}")
    print(f"Mean latency: {statistics.mean(latencies):.2f} ms")
    print(f"P95 latency: {statistics.quantiles(latencies, n=20)[18]:.2f} ms")
    print(f"Max latency: {max(latencies):.2f} ms")


if __name__ == "__main__":
    run_benchmark()
