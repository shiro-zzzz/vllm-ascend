#!/usr/bin/env python3
"""
Benchmark script for comparing EXEC_NPU_CMD vs EXEC_NPU_CMD_V2
on the get_dispatch_layout operator.

Measures end-to-end performance under TASK_QUEUE_ENABLE=1 and
TASK_QUEUE_ENABLE=2 to quantify the scheduling improvement of the V2
calling convention.

Usage:
    # Run with TASK_QUEUE_ENABLE=1 (default)
    TASK_QUEUE_ENABLE=1 python benchmarks/ops/bench_dispatch_layout.py

    # Run with TASK_QUEUE_ENABLE=2
    TASK_QUEUE_ENABLE=2 python benchmarks/ops/bench_dispatch_layout.py

    # Sweep both modes automatically
    python benchmarks/ops/bench_dispatch_layout.py --sweep
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def bench(fn, num_warmups: int = 50, num_tests: int = 100):
    """Per-op latency benchmark: synchronize between each iteration.

    Returns (avg, min, max) in seconds.
    """
    device = torch.device("npu")
    torch.npu.synchronize()

    # Flush L2 cache with 256 MB
    cache = torch.empty(int(256e6 // 4), dtype=torch.int32, device=device)

    # Warmup
    for _ in range(num_warmups):
        fn()

    cache.zero_()
    torch.npu.synchronize()

    # Timing – sync per iteration to measure single-op latency
    times = []
    for _ in range(num_tests):
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        torch.npu.synchronize()
        elapsed = start.elapsed_time(end) / 1e3  # ms -> s
        times.append(elapsed)

    times = np.array(times[1:])  # drop first
    return np.average(times), np.min(times), np.max(times)


def bench_async(fn, num_warmups: int = 50, num_tests: int = 100):
    """Pipeline throughput benchmark: synchronize only outside the loop.

    All iterations are submitted back-to-back without intermediate syncs,
    then a single sync is performed at the end.  This measures the
    amortised host-side submission cost and pipeline throughput, which
    is more representative of real inference workloads.

    Returns (avg, min, max) in seconds – avg is total_time / num_tests.
    min and max are identical to avg here (single measurement window).
    """
    device = torch.device("npu")
    torch.npu.synchronize()

    # Flush L2 cache with 256 MB
    cache = torch.empty(int(256e6 // 4), dtype=torch.int32, device=device)

    # Warmup
    for _ in range(num_warmups):
        fn()

    cache.zero_()
    torch.npu.synchronize()

    # Timing – sync only before and after the entire loop
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for _ in range(num_tests):
        fn()
    end.record()

    torch.npu.synchronize()
    total_elapsed = start.elapsed_time(end) / 1e3  # ms -> s
    avg = total_elapsed / num_tests
    return avg, avg, avg


def run_benchmark(args):
    """Run the benchmark for a single TASK_QUEUE_ENABLE setting."""
    task_queue_mode = os.environ.get("TASK_QUEUE_ENABLE", "1")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.manual_seed(42)

    num_tokens_list = args.num_tokens
    num_experts = args.num_experts
    num_topk = args.num_topk
    num_ranks = args.num_ranks
    num_warmups = args.num_warmups
    num_tests = args.num_tests

    # ---- Helper: run one benchmark round ----
    def _run_round(bench_fn, bench_name):
        print(f"\n{'=' * 80}")
        print(f"  Dispatch Layout Benchmark  |  TASK_QUEUE_ENABLE={task_queue_mode}")
        print(f"  Mode: {bench_name}")
        print(f"  num_experts={num_experts}, num_topk={num_topk}, num_ranks={num_ranks}")
        print(f"  warmups={num_warmups}, iterations={num_tests}")
        print(f"{'=' * 80}")
        print(
            f"{'num_tokens':>12s} | "
            f"{'EXEC_NPU_CMD avg(us)':>22s} | "
            f"{'EXEC_NPU_CMD_V2 avg(us)':>25s} | "
            f"{'speedup':>8s}"
        )
        print("-" * 80)

        last_topk_idx = None
        for num_tokens in num_tokens_list:
            # Generate routing data
            scores = torch.randn(
                (num_tokens, num_experts), dtype=torch.float32, device=device
            ).abs() + 1
            topk_idx = torch.topk(
                scores, num_topk, dim=-1, largest=True, sorted=False
            )[1]
            last_topk_idx = topk_idx

            # ---- Benchmark get_dispatch_layout (EXEC_NPU_CMD) ----
            t_old_avg, _, _ = bench_fn(
                lambda: torch.ops._C_ascend.get_dispatch_layout(
                    topk_idx, num_experts, num_ranks
                ),
                num_warmups=num_warmups,
                num_tests=num_tests,
            )

            # ---- Benchmark get_dispatch_layout_v2 (EXEC_NPU_CMD_V2) ----
            t_new_avg, _, _ = bench_fn(
                lambda: torch.ops._C_ascend.get_dispatch_layout_v2(
                    topk_idx, num_experts, num_ranks
                ),
                num_warmups=num_warmups,
                num_tests=num_tests,
            )

            speedup = t_old_avg / t_new_avg if t_new_avg > 0 else float("inf")
            print(
                f"{num_tokens:>12d} | "
                f"{t_old_avg * 1e6:>22.2f} | "
                f"{t_new_avg * 1e6:>25.2f} | "
                f"{speedup:>7.3f}x"
            )

        # Correctness check with last config
        if last_topk_idx is not None:
            ref_expert, ref_idx = torch.ops._C_ascend.get_dispatch_layout(
                last_topk_idx, num_experts, num_ranks
            )
            v2_expert, v2_idx = torch.ops._C_ascend.get_dispatch_layout_v2(
                last_topk_idx, num_experts, num_ranks
            )
            expert_match = torch.equal(ref_expert, v2_expert)
            idx_match = torch.equal(ref_idx, v2_idx)
            status = "PASS" if (expert_match and idx_match) else "FAIL"
            print(f"\nCorrectness check: {status}")
            if not expert_match:
                print("  WARNING: num_tokens_per_expert mismatch!")
            if not idx_match:
                print("  WARNING: send_token_idx_small mismatch!")
        print()

    # ---- Round 1: per-op latency (sync per iteration) ----
    _run_round(bench, "per-op latency (sync per iteration)")

    # ---- Round 2: pipeline throughput (sync outside loop) ----
    _run_round(bench_async, "pipeline throughput (sync outside loop)")


def sweep_modes(args):
    """Run the benchmark under both TASK_QUEUE_ENABLE=1 and 2."""
    script = os.path.abspath(__file__)
    base_cmd = [sys.executable, script]
    # Forward relevant args
    base_cmd += ["--num-experts", str(args.num_experts)]
    base_cmd += ["--num-topk", str(args.num_topk)]
    base_cmd += ["--num-ranks", str(args.num_ranks)]
    base_cmd += ["--num-warmups", str(args.num_warmups)]
    base_cmd += ["--num-tests", str(args.num_tests)]
    for nt in args.num_tokens:
        base_cmd += ["--num-tokens", str(nt)]

    for mode in [1, 2]:
        print(f"\n>>> Launching with TASK_QUEUE_ENABLE={mode} <<<")
        env = os.environ.copy()
        env["TASK_QUEUE_ENABLE"] = str(mode)
        subprocess.run(base_cmd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark EXEC_NPU_CMD vs EXEC_NPU_CMD_V2 on get_dispatch_layout"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048, 4096],
        help="List of num_tokens values to benchmark (default: 64..4096)",
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256)"
    )
    parser.add_argument(
        "--num-topk",
        type=int,
        default=8,
        help="Number of top-k experts per token (default: 8)",
    )
    parser.add_argument(
        "--num-ranks",
        type=int,
        default=8,
        help="Number of ranks (default: 8)",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=50,
        help="Warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=100,
        help="Benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Automatically run under both TASK_QUEUE_ENABLE=1 and 2",
    )
    args = parser.parse_args()

    if args.sweep:
        sweep_modes(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
