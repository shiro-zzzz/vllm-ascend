#!/usr/bin/env python3
"""
Benchmark script for comparing EXEC_NPU_CMD vs EXEC_NPU_CMD_V2
on the get_dispatch_layout operator.

Measures end-to-end performance and kernel-level performance under
TASK_QUEUE_ENABLE=1 and TASK_QUEUE_ENABLE=2 to quantify the scheduling
improvement of the V2 calling convention.

Usage:
    # Run with TASK_QUEUE_ENABLE=1 (default)
    TASK_QUEUE_ENABLE=1 python benchmarks/ops/bench_dispatch_layout.py

    # Run with TASK_QUEUE_ENABLE=2
    TASK_QUEUE_ENABLE=2 python benchmarks/ops/bench_dispatch_layout.py

    # Sweep both modes automatically
    python benchmarks/ops/bench_dispatch_layout.py --sweep

    # Specify kernel name for profiling (if auto-detect fails)
    python benchmarks/ops/bench_dispatch_layout.py --kernel-name aclnnDispatchLayout
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Union

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


def _cpu_busy_work(duration_us: float, mat_size: int = 64):
    """Simulate CPU-side scheduling/compute work for *duration_us* microseconds.

    Uses small CPU matrix multiplications to create realistic load that
    cannot be optimized away.  This mimics the kind of work a real
    inference scheduler does between operator submissions (attention mask
    construction, token bookkeeping, Python scheduling logic, etc.).
    """
    if duration_us <= 0:
        return
    a = torch.randn(mat_size, mat_size, device="cpu")
    b = torch.randn(mat_size, mat_size, device="cpu")
    deadline = time.perf_counter() + duration_us * 1e-6
    while time.perf_counter() < deadline:
        # Small CPU matmul – realistic workload, cannot be elided
        torch.mm(a, b)


def bench_busy(
    fn,
    cpu_busy_us: float = 100.0,
    num_warmups: int = 50,
    num_tests: int = 100,
):
    """Pipeline throughput with simulated main-thread CPU load.

    Between each operator submission, the main thread performs CPU-side
    work for approximately *cpu_busy_us* microseconds.  This simulates
    real inference scheduling overhead and reveals the pipeline benefit
    of TASK_QUEUE_ENABLE=2 + EXEC_NPU_CMD_V2:

      - V1 / V2 mode 1: CPU work and NPU ConvertTypes/GetWorkspaceSize
        both happen on the main thread serially.
      - V2 mode 2: ConvertTypes/GetWorkspaceSize are deferred to the
        task-queue thread, so they overlap with the CPU work → better
        overall throughput.

    Returns (avg, min, max) in seconds.
    """
    device = torch.device("npu")
    torch.npu.synchronize()

    cache = torch.empty(int(256e6 // 4), dtype=torch.int32, device=device)

    for _ in range(num_warmups):
        fn()
        _cpu_busy_work(cpu_busy_us)

    cache.zero_()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for _ in range(num_tests):
        fn()
        _cpu_busy_work(cpu_busy_us)
    end.record()

    torch.npu.synchronize()
    total_elapsed = start.elapsed_time(end) / 1e3  # ms -> s
    avg = total_elapsed / num_tests
    return avg, avg, avg


def bench_kineto(
    fn,
    kernel_names: Union[str, tuple],
    num_warmups: int = 50,
    num_tests: int = 30,
    suppress_kineto_output: bool = True,
):
    """Profile with torch_npu.profiler and extract NPU kernel durations.

    Args:
        fn: callable to benchmark.
        kernel_names: exact kernel name(s) to look for in the trace.
        num_warmups: warmup iterations (run before profiling).
        num_tests: iterations inside the profiling active window.
        suppress_kineto_output: redirect profiler stdout/stderr to devnull.

    Returns:
        If kernel_names is a str  -> float (avg kernel duration in seconds).
        If kernel_names is a tuple -> list of floats.
    """
    device = torch.device("npu")

    # Warmup outside profiler
    for _ in range(num_warmups):
        fn()
    torch.npu.synchronize()

    # Redirect profiler logs if requested
    class _SuppressOutput:
        def __enter__(self):
            if suppress_kineto_output:
                self._out = open(os.devnull, "w")
                self._err = open(os.devnull, "w")
                self._old_stdout_fd = os.dup(sys.stdout.fileno())
                self._old_stderr_fd = os.dup(sys.stderr.fileno())
                os.dup2(self._out.fileno(), sys.stdout.fileno())
                os.dup2(self._err.fileno(), sys.stderr.fileno())
            return self

        def __exit__(self, *_):
            if suppress_kineto_output:
                os.dup2(self._old_stdout_fd, sys.stdout.fileno())
                os.dup2(self._old_stderr_fd, sys.stderr.fileno())
                os.close(self._old_stdout_fd)
                os.close(self._old_stderr_fd)
                self._out.close()
                self._err.close()

    with _SuppressOutput():
        schedule = torch_npu.profiler.schedule(
            wait=1, warmup=0, active=1, repeat=1
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=schedule,
        ) as prof:
            for step_idx in range(2):
                for _ in range(num_tests):
                    fn()
                torch.npu.synchronize()
                prof.step()

    # Export trace to temp JSON
    temp_path = Path(tempfile.gettempdir()) / f"trace_{uuid.uuid4().hex}.json"
    prof.export_chrome_trace(str(temp_path))
    raw_data = json.loads(temp_path.read_text())
    os.unlink(temp_path)

    # Chrome trace format: {"traceEvents": [...]} or flat list
    if isinstance(raw_data, dict):
        profile_data = raw_data.get("traceEvents", [])
    else:
        profile_data = raw_data

    # Parse kernel durations
    is_tuple = isinstance(kernel_names, tuple)
    names = (kernel_names,) if not is_tuple else kernel_names

    kernel_durations = []
    for kname in names:
        events = [
            e for e in profile_data
            if kname == e.get("name") and "dur" in e
        ]
        if not events:
            # Fuzzy match: find events containing the kernel name
            events = [
                e for e in profile_data
                if kname in e.get("name", "") and "dur" in e
            ]
        if not events:
            print(f"  [kineto] WARNING: kernel '{kname}' not found in trace")
            kernel_durations.append(0.0)
            continue
        events = sorted(events, key=lambda e: e["ts"])
        durations = [e["dur"] / 1e6 for e in events]  # us -> s
        avg_dur = sum(durations) / len(durations)
        kernel_durations.append(avg_dur)

    return kernel_durations if is_tuple else kernel_durations[0]


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
    kernel_name = args.kernel_name

    # ---- Helper: run one e2e benchmark round ----
    def _run_e2e_round(bench_fn, bench_name):
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
            scores = torch.randn(
                (num_tokens, num_experts), dtype=torch.float32, device=device
            ).abs() + 1
            topk_idx = torch.topk(
                scores, num_topk, dim=-1, largest=True, sorted=False
            )[1]
            last_topk_idx = topk_idx

            t_old_avg, _, _ = bench_fn(
                lambda: torch.ops._C_ascend.get_dispatch_layout(
                    topk_idx, num_experts, num_ranks
                ),
                num_warmups=num_warmups,
                num_tests=num_tests,
            )

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

        return last_topk_idx

    # ---- Helper: run one kineto profiling round ----
    def _run_kineto_round():
        print(f"\n{'=' * 80}")
        print(f"  Dispatch Layout Benchmark  |  TASK_QUEUE_ENABLE={task_queue_mode}")
        print(f"  Mode: kernel profiling (torch_npu.profiler / kineto)")
        print(f"  kernel_name={kernel_name}")
        print(f"  num_experts={num_experts}, num_topk={num_topk}, num_ranks={num_ranks}")
        print(f"  warmups={num_warmups}, profiling iterations={num_tests}")
        print(f"{'=' * 80}")
        print(
            f"{'num_tokens':>12s} | "
            f"{'CMD e2e(us)':>14s} | "
            f"{'CMD kernel(us)':>16s} | "
            f"{'V2 e2e(us)':>14s} | "
            f"{'V2 kernel(us)':>16s} | "
            f"{'e2e spdup':>10s} | "
            f"{'kern spdup':>10s}"
        )
        print("-" * 105)

        for num_tokens in num_tokens_list:
            scores = torch.randn(
                (num_tokens, num_experts), dtype=torch.float32, device=device
            ).abs() + 1
            topk_idx = torch.topk(
                scores, num_topk, dim=-1, largest=True, sorted=False
            )[1]

            # e2e with bench_async (pipeline mode, most relevant)
            t_old_e2e, _, _ = bench_async(
                lambda: torch.ops._C_ascend.get_dispatch_layout(
                    topk_idx, num_experts, num_ranks
                ),
                num_warmups=num_warmups,
                num_tests=num_tests,
            )

            t_new_e2e, _, _ = bench_async(
                lambda: torch.ops._C_ascend.get_dispatch_layout_v2(
                    topk_idx, num_experts, num_ranks
                ),
                num_warmups=num_warmups,
                num_tests=num_tests,
            )

            # kernel time via kineto
            t_old_kern = bench_kineto(
                lambda: torch.ops._C_ascend.get_dispatch_layout(
                    topk_idx, num_experts, num_ranks
                ),
                kernel_names=kernel_name,
                num_warmups=num_warmups,
                num_tests=num_tests,
            )

            t_new_kern = bench_kineto(
                lambda: torch.ops._C_ascend.get_dispatch_layout_v2(
                    topk_idx, num_experts, num_ranks
                ),
                kernel_names=kernel_name,
                num_warmups=num_warmups,
                num_tests=num_tests,
            )

            e2e_speedup = (
                t_old_e2e / t_new_e2e if t_new_e2e > 0 else float("inf")
            )
            kern_speedup = (
                t_old_kern / t_new_kern
                if t_old_kern > 0 and t_new_kern > 0
                else float("nan")
            )
            print(
                f"{num_tokens:>12d} | "
                f"{t_old_e2e * 1e6:>14.2f} | "
                f"{t_old_kern * 1e6:>16.2f} | "
                f"{t_new_e2e * 1e6:>14.2f} | "
                f"{t_new_kern * 1e6:>16.2f} | "
                f"{e2e_speedup:>9.3f}x | "
                f"{kern_speedup:>9.3f}x"
            )
        print()

    # ========== Round 1: per-op latency (sync per iteration) ==========
    last_topk_idx = _run_e2e_round(
        bench, "per-op latency (sync per iteration)"
    )

    # ========== Round 2: pipeline throughput (sync outside loop) ==========
    _run_e2e_round(bench_async, "pipeline throughput (sync outside loop)")

    # ========== Round 3: busy main thread (simulate inference scheduling) ====
    cpu_busy_us = args.cpu_busy_us
    if cpu_busy_us > 0:
        _run_e2e_round(
            lambda fn, **kw: bench_busy(fn, cpu_busy_us=cpu_busy_us, **kw),
            f"busy main thread (cpu_busy={cpu_busy_us}us per iter)",
        )

    # ========== Round 4: kernel profiling (kineto) ==========
    # _run_kineto_round()

    # ========== Correctness check ==========
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
        print(f"Correctness check: {status}")
        if not expert_match:
            print("  WARNING: num_tokens_per_expert mismatch!")
        if not idx_match:
            print("  WARNING: send_token_idx_small mismatch!")
    print()


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
    base_cmd += ["--cpu-busy-us", str(args.cpu_busy_us)]
    base_cmd += ["--kernel-name", args.kernel_name]

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
        default=10000,
        help="Benchmark iterations (default: 10000)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Automatically run under both TASK_QUEUE_ENABLE=1 and 2",
    )
    parser.add_argument(
        "--cpu-busy-us",
        type=float,
        default=100.0,
        help="Simulated CPU busy time (us) per iteration for busy-mode round "
             "(default: 100). Set to 0 to skip this round.",
    )
    parser.add_argument(
        "--kernel-name",
        type=str,
        default="aclnnDispatchLayout",
        help="Kernel name to match in profiler trace (default: aclnnDispatchLayout)",
    )
    args = parser.parse_args()

    if args.sweep:
        sweep_modes(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
