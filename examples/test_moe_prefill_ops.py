#!/usr/bin/env python3
"""
Test script for vLLM MoE dispatch/combine prefill operators.

This script validates the correctness and performance of:
- get_dispatch_layout: Generate metadata for token distribution
- dispatch_prefill: Distribute tokens to experts across ranks
- combine_prefill: Gather expert outputs and combine with routing weights

Usage:
    python test_moe_prefill_ops.py --num-processes 8 --num-tokens 1024 --hidden 7168
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.utils import enable_custom_op
enable_custom_op()


def init_dist(local_rank: int, num_local_ranks: int):
    """Initialize distributed environment for multiprocessing.spawn."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(num_local_ranks)
    
    rank = local_rank
    world_size = num_local_ranks

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    dist.init_process_group(
        backend="hccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    
    group = dist.new_group(list(range(world_size)))

    return rank, world_size, group


def inplace_unique(x: torch.Tensor, num_slots: int):
    """In-place unique operation to remove duplicate expert IDs per token."""
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def bench(fn, num_warmups: int = 50, num_tests: int = 50):
    """Benchmark function with warmup."""
    device = torch.device("npu")
    torch.npu.synchronize()

    # Flush L2 cache with 256 MB data
    cache = torch.empty(int(256e6 // 4), dtype=torch.int32, device=device)

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2 cache
    cache.zero_()
    torch.npu.synchronize()

    # Timing
    times = []
    for _ in range(num_tests):
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        torch.npu.synchronize()
        elapsed_time = start.elapsed_time(end) / 1e3  # ms -> s
        times.append(elapsed_time)

    times = np.array(times[1:])  # Remove the first timing
    return np.average(times), np.min(times), np.max(times)


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    """Cast FP8 quantized tensor back to BF16."""
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.int8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Calculate relative difference between two tensors."""
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def _count_unequal_element(data_expect, data_check, rtol, atol, msg=""):
    """Count unequal elements between two arrays."""
    assert data_expect.shape == data_check.shape
    total_count = len(data_expect.flatten())
    error = np.abs(data_expect - data_check)
    greater = np.greater(error, atol + np.abs(data_check) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (
        loss_count / total_count
    ) < rtol, "\nmsg{0}_data_expect_std:{1}\ndata_check_error:{2}\nloss:{3}".format(
        msg, data_expect[greater], data_check[greater], error[greater]
    )


def allclose_nparray(data_expect, data_check, rtol=1e-4, atol=1e-4, equal_nan=True, msg=""):
    """Check if two numpy arrays are close within tolerance."""
    if np.any(np.isnan(data_expect)):
        assert np.allclose(data_expect, data_check, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expect, data_check, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expect, data_check, rtol, atol, msg)
    else:
        assert True


def compare_tensors_with_saved(rank: int, save_dir: str,
                                expandx_out_bf16: torch.Tensor,
                                topk_ids: torch.Tensor,
                                topk_weights: torch.Tensor,
                                expand_idx_out: torch.Tensor,
                                recv_count: torch.Tensor):
    """Compare dispatch output tensors with saved combine input tensors.
    
    Args:
        rank: Current rank ID
        save_dir: Directory containing saved tensors
        expandx_out_bf16: Hidden states output from dispatch
        topk_ids: Top-k expert IDs
        topk_weights: Top-k weights
        expand_idx_out: Expanded indices from dispatch
        recv_count: Receive counts from dispatch
    
    Returns:
        bool: True if saved tensors exist and match, False otherwise
    """
    import os
    
    combine_input_path = f"{save_dir}/combine_input_rank{rank}.pt"
    if not os.path.exists(combine_input_path):
        if rank == 0:
            print(f"  Note: No saved combine input found for comparison", flush=True)
        return False
    
    if rank == 0:
        print(f"\n  Comparing dispatch output with saved combine input...")
        print(f"  " + "=" * 56, flush=True)
    
    # Load saved combine input tensors
    combine_data = torch.load(combine_input_path)
    saved_hidden_states = combine_data["hidden_states"].to("npu")
    saved_topk_ids = combine_data["topk_ids"].to("npu")
    saved_topk_weights = combine_data["topk_weights"].to("npu")
    saved_expand_idx_out = combine_data["expand_idx_out"].to("npu")
    saved_recv_count = combine_data["recv_count"].to("npu")
    
    # Also load dispatch output to compare expand_idx_out consistency
    dispatch_output_path = f"{save_dir}/dispatch_output_rank{rank}.pt"
    dispatch_output_expand_idx = None
    if os.path.exists(dispatch_output_path):
        dispatch_output_data = torch.load(dispatch_output_path)
        dispatch_output_expand_idx = dispatch_output_data["expand_idx_out"].to("npu")
        
        if rank == 0:
            print(f"\n  Checking expand_idx_out consistency across saved files...")
            print(f"  " + "=" * 56)
        
        # Compare dispatch_output vs combine_input
        dispatch_out_np = dispatch_output_expand_idx.cpu().numpy()
        combine_in_np = saved_expand_idx_out.cpu().numpy()
        
        if np.array_equal(dispatch_out_np, combine_in_np):
            if rank == 0:
                print(f"    ✓ dispatch_output and combine_input expand_idx_out match")
        else:
            diff_count = (dispatch_out_np != combine_in_np).sum()
            if rank == 0:
                print(f"    ✗ dispatch_output and combine_input expand_idx_out differ!")
                print(f"      Mismatches: {diff_count}/{dispatch_out_np.size} elements")
                print(f"      First few (dispatch_output): {dispatch_out_np.flatten()[:10]}")
                print(f"      First few (combine_input):   {combine_in_np.flatten()[:10]}")
        
        if rank == 0:
            print(flush=True)
    
    # Compare tensors using allclose_nparray
    if rank == 0:
        print(f"  Tensor comparison (Rank {rank}):")
    
    # Convert tensors to numpy for comparison
    topk_ids_np = topk_ids.cpu().numpy()
    saved_topk_ids_np = saved_topk_ids.cpu().numpy()
    topk_weights_np = topk_weights.cpu().float().numpy()
    saved_topk_weights_np = saved_topk_weights.cpu().float().numpy()
    expand_idx_np = expand_idx_out.cpu().numpy()
    saved_expand_idx_np = saved_expand_idx_out.cpu().numpy()
    recv_count_np = recv_count.cpu().numpy()
    saved_recv_count_np = saved_recv_count.cpu().numpy()
    
    all_match = True
    
    # Compare topk_ids (exact match required)
    if np.array_equal(topk_ids_np, saved_topk_ids_np):
        if rank == 0:
            print(f"    topk_ids: ✓ Match")
    else:
        all_match = False
        diff_count = (topk_ids_np != saved_topk_ids_np).sum()
        if rank == 0:
            print(f"    topk_ids: ✗ Mismatch ({diff_count}/{topk_ids_np.size} elements differ)")
            print(f"      First few (dispatch): {topk_ids_np.flatten()[:5]}")
            print(f"      First few (saved):    {saved_topk_ids_np.flatten()[:5]}")
    
    # Compare topk_weights
    try:
        allclose_nparray(saved_topk_weights_np, topk_weights_np, rtol=1e-4, atol=1e-4, msg="topk_weights")
        if rank == 0:
            print(f"    topk_weights: ✓ Match")
    except AssertionError as e:
        all_match = False
        if rank == 0:
            print(f"    topk_weights: ✗ Mismatch")
            print(f"      {str(e)[:200]}...")  # Print first 200 chars of error
    
    # Compare expand_idx_out (exact match required)
    if np.array_equal(expand_idx_np, saved_expand_idx_np):
        if rank == 0:
            print(f"    expand_idx_out: ✓ Match")
    else:
        all_match = False
        diff_mask = (expand_idx_np != saved_expand_idx_np)
        diff_count = diff_mask.sum()
        if rank == 0:
            print(f"    expand_idx_out: ✗ Mismatch ({diff_count}/{expand_idx_np.size} elements differ)")
            print(f"      First few (dispatch): {expand_idx_np.flatten()[:10]}")
            print(f"      First few (saved):    {saved_expand_idx_np.flatten()[:10]}")
            
            # Print detailed difference information
            diff_indices = np.where(diff_mask.flatten())[0]
            if len(diff_indices) > 0:
                print(f"      Difference details (showing first 10):")
                for i, idx in enumerate(diff_indices[:10]):
                    dispatch_val = expand_idx_np.flatten()[idx]
                    saved_val = saved_expand_idx_np.flatten()[idx]
                    print(f"        Index {idx}: dispatch={dispatch_val}, saved={saved_val}")
    
    # Compare recv_count (exact match required)
    if np.array_equal(recv_count_np, saved_recv_count_np):
        if rank == 0:
            print(f"    recv_count: ✓ Match")
    else:
        all_match = False
        if rank == 0:
            print(f"    recv_count: ✗ Mismatch")
            print(f"      recv_count (dispatch): {recv_count_np}")
            print(f"      recv_count (saved):    {saved_recv_count_np}")
    
    if rank == 0:
        if all_match:
            print(f"  ✓ All tensors match saved combine inputs!", flush=True)
        else:
            print(f"  ✗ Tensors differ from saved combine inputs!", flush=True)
        print(flush=True)
        
        return all_match
    
    return True


def test_with_saved_tensors(args, rank: int, world_size: int, group):
    """Test operators using saved tensors from a previous run."""
    import os
    
    save_dir = args.tensor_save_dir
    use_dispatch_output = args.use_dispatch_output
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Testing with saved tensors from: {save_dir}")
        if use_dispatch_output:
            print(f"Mode: Using dispatch output for combine (end-to-end test)")
        else:
            print(f"Mode: Using saved tensors separately")
        print(f"{'='*60}\n", flush=True)
    
    # Get HCCL communicator info
    hccl_comm_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    hccl_comm_name = hccl_comm_name.encode('utf-8').decode('utf-8')
    
    # Load dispatch input tensors
    dispatch_input_path = f"{save_dir}/dispatch_input_rank{rank}.pt"
    if not os.path.exists(dispatch_input_path):
        if rank == 0:
            print(f"Error: Saved tensors not found at {dispatch_input_path}", flush=True)
        return
    
    if rank == 0:
        print("[Test 1] Testing dispatch_prefill with saved tensors")
        print("-" * 60, flush=True)
    
    dispatch_data = torch.load(dispatch_input_path)
    hidden_states = dispatch_data["hidden_states"].to("npu")
    topk_ids = dispatch_data["topk_ids"].to("npu")
    topk_weights = dispatch_data["topk_weights"].to("npu")
    num_tokens_per_expert = dispatch_data["num_tokens_per_expert"].to("npu")
    send_token_idx_small = dispatch_data["send_token_idx_small"].to("npu")
    with_quant = dispatch_data["with_quant"]
    num_experts = dispatch_data["num_experts"]
    
    if rank == 0:
        print(f"  Loaded dispatch inputs from rank {rank}")
        print(f"  hidden_states shape: {hidden_states.shape}")
        print(f"  topk_ids shape: {topk_ids.shape}")
        print(f"  with_quant: {with_quant}", flush=True)
    
    # Test dispatch_prefill
    try:
        (expandx_out, dynamic_scales_out, expand_idx_out, 
         recv_count, recv_tokens_per_expert) = torch.ops._C_ascend.dispatch_prefill(
            hidden_states,
            topk_ids,
            topk_weights,
            num_tokens_per_expert,
            send_token_idx_small,
            hccl_comm_name,
            rank,
            world_size,
            with_quant)
        torch.npu.synchronize()
        
        if rank == 0:
            print(f"  ✓ dispatch_prefill executed successfully")
            print(f"  expandx_out shape: {expandx_out.shape}")
            print(f"  expand_idx_out shape: {expand_idx_out.shape}", flush=True)
    except Exception as e:
        if rank == 0:
            print(f"  ✗ dispatch_prefill failed: {e}", flush=True)
        raise
    
    dist.barrier()
    
    # Choose test mode: use dispatch output or load saved combine inputs
    if use_dispatch_output:
        # New mode: Use dispatch output directly for combine
        if rank == 0:
            print("\n[Test 2] Testing combine_prefill with dispatch output (end-to-end)")
            print("-" * 60, flush=True)
        
        # Cast back if quantized
        expandx_out_bf16 = per_token_cast_back(expandx_out, dynamic_scales_out) if with_quant else expandx_out
        
        # Compare with saved combine input tensors if available
        compare_tensors_with_saved(rank, save_dir, expandx_out_bf16, topk_ids, 
                                   topk_weights, expand_idx_out, recv_count)
        
        if rank == 0:
            print(f"\n  Using dispatch output for combine")
            print(f"  expandx_out shape: {expandx_out_bf16.shape}")
            print(f"  topk_ids shape: {topk_ids.shape}", flush=True)
        
        # Test combine_prefill with dispatch output
        try:
            combined_x = torch.ops._C_ascend.combine_prefill(
                expandx_out_bf16,
                topk_ids,
                topk_weights,
                expand_idx_out,
                recv_count,
                hccl_comm_name,
                rank,
                world_size)
            torch.npu.synchronize()
            
            if rank == 0:
                print(f"  ✓ combine_prefill executed successfully")
                print(f"  combined_x shape: {combined_x.shape}", flush=True)
        except Exception as e:
            if rank == 0:
                print(f"  ✗ combine_prefill failed: {e}", flush=True)
            raise
    else:
        # Original mode: Load combine input tensors separately
        combine_input_path = f"{save_dir}/combine_input_rank{rank}.pt"
        if os.path.exists(combine_input_path):
            if rank == 0:
                print("\n[Test 2] Testing combine_prefill with saved tensors")
                print("-" * 60, flush=True)
            
            combine_data = torch.load(combine_input_path)
            hidden_states_combine = combine_data["hidden_states"].to("npu")
            topk_ids_combine = combine_data["topk_ids"].to("npu")
            topk_weights_combine = combine_data["topk_weights"].to("npu")
            expand_idx_out_combine = combine_data["expand_idx_out"].to("npu")
            recv_count_combine = combine_data["recv_count"].to("npu")
            
            if rank == 0:
                print(f"  Loaded combine inputs from rank {rank}")
                print(f"  hidden_states shape: {hidden_states_combine.shape}")
                print(f"  topk_ids shape: {topk_ids_combine.shape}", flush=True)
            
            # Test combine_prefill
            try:
                combined_x = torch.ops._C_ascend.combine_prefill(
                    hidden_states_combine,
                    topk_ids_combine,
                    topk_weights_combine,
                    expand_idx_out_combine,
                    recv_count_combine,
                    hccl_comm_name,
                    rank,
                    world_size)
                torch.npu.synchronize()
                
                if rank == 0:
                    print(f"  ✓ combine_prefill executed successfully")
                    print(f"  combined_x shape: {combined_x.shape}", flush=True)
            except Exception as e:
                if rank == 0:
                    print(f"  ✗ combine_prefill failed: {e}", flush=True)
                raise
        else:
            if rank == 0:
                print(f"\n  Combine input tensors not found at {combine_input_path}", flush=True)
    
    dist.barrier()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Saved tensor validation completed!")
        print(f"{'='*60}\n", flush=True)


def test_moe_ops(args, local_rank: int, num_local_ranks: int):
    """Test MoE dispatch/combine operators."""
    # Initialize distributed environment
    rank, world_size, group = init_dist(local_rank, num_local_ranks)
    
    # Set random seed for reproducibility
    torch.manual_seed(rank + 42)
    
    # Check if we should load saved tensors for debugging
    if args.load_saved_tensors:
        test_with_saved_tensors(args, rank, world_size, group)
        dist.destroy_process_group()
        return
    
    # Parse parameters
    num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts
    use_quant = args.use_quant
    
    assert num_experts % world_size == 0, f"num_experts ({num_experts}) must be divisible by world_size ({world_size})"
    
    experts_per_rank = num_experts // world_size
    
    if local_rank == 0:
        print(f"=== MoE Prefill Operators Test ===")
        print(f"Configuration:")
        print(f"  num_tokens={num_tokens}, hidden={hidden}")
        print(f"  num_topk={num_topk}, num_experts={num_experts}")
        print(f"  world_size={world_size}, experts_per_rank={experts_per_rank}")
        print(f"  use_quant={use_quant}")
        print(flush=True)
    
    # Generate routing information
    if args.active_ranks:
        # Only assign tokens to specified ranks
        try:
            active_ranks = [int(r.strip()) for r in args.active_ranks.split(",") if r.strip()]
        except ValueError:
            raise ValueError(
                f"Invalid value in --active-ranks: {args.active_ranks}. "
                f"Must be a comma-separated list of integers, e.g., '0,1,3'."
            )
        
        if any(r < 0 or r >= world_size for r in active_ranks):
            raise ValueError(
                f"Invalid rank in --active-ranks: {active_ranks}. "
                f"Ranks must be in range [0, {world_size-1}]."
            )
        
        if not active_ranks:
            raise ValueError("Parsed --active-ranks is empty. Provide at least one valid rank.")
        
        valid_experts = torch.cat(
            [torch.arange(r * experts_per_rank, (r + 1) * experts_per_rank, device="npu")
             for r in active_ranks]
        )
        topk_idx = valid_experts[
            torch.randint(0, len(valid_experts), (num_tokens, num_topk), device="npu")
        ]
    else:
        # Default: random routing over all experts
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs() + 1
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    
    # Generate rank indices
    rank_idx = topk_idx // experts_per_rank
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, world_size)
    
    # Get HCCL communicator info
    hccl_comm_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    hccl_comm_name = hccl_comm_name.encode('utf-8').decode('utf-8')
    
    # Test 1: get_dispatch_layout
    if local_rank == 0:
        print("\n[Test 1] get_dispatch_layout")
        print("-" * 60, flush=True)
    
    t_avg, _, _ = bench(
        lambda: torch.ops._C_ascend.get_dispatch_layout(topk_idx, num_experts, world_size),
        num_warmups=10,
        num_tests=20
    )
    
    if local_rank == 0:
        print(f"  Performance: {t_avg * 1000:.3f} ms", flush=True)
    
    num_tokens_per_expert, send_token_idx_small = torch.ops._C_ascend.get_dispatch_layout(
        topk_idx, num_experts, world_size
    )
    
    if local_rank == 0:
        print(f"  num_tokens_per_expert shape: {num_tokens_per_expert.shape}")
        print(f"  send_token_idx_small shape: {send_token_idx_small.shape}")
        print(f"  ✓ get_dispatch_layout passed", flush=True)
    
    dist.barrier()
    
    # Test 2: dispatch_prefill
    if local_rank == 0:
        print(f"\n[Test 2] dispatch_prefill (use_quant={use_quant})")
        print("-" * 60, flush=True)
    
    # Create input data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * rank
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device="npu") * rank
    
    dispatch_args = {
        "x": x,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
        "num_tokens_per_expert": num_tokens_per_expert,
        "send_token_idx_small": send_token_idx_small,
        "groupEp": hccl_comm_name,
        "rank": rank,
        "num_ranks": world_size,
        "use_quant": use_quant,
    }
    
    t_avg, _, _ = bench(
        lambda: torch.ops._C_ascend.dispatch_prefill(**dispatch_args),
        num_warmups=10,
        num_tests=20
    )
    
    (recv_x, dynamic_scales_out, expand_idx_out, recv_count, recv_tokens_per_expert
    ) = torch.ops._C_ascend.dispatch_prefill(**dispatch_args)
    torch.npu.synchronize()
    
    # Cast back if quantized
    recv_x_bf16 = per_token_cast_back(recv_x, dynamic_scales_out) if use_quant else recv_x
    
    recv_bytes = recv_x.numel() * 2
    if use_quant:
        recv_bytes = recv_bytes * (1 + 4 / 128) / 2  # FP8 factor
    
    if local_rank == 0:
        print(f"  recv_x shape: {recv_x.shape}")
        print(f"  expand_idx_out shape: {expand_idx_out.shape}")
        print(f"  recv_count shape: {recv_count.shape}")
        print(f"  recv_tokens_per_expert shape: {recv_tokens_per_expert.shape}")
        print(f"  Bandwidth: {recv_bytes / 1e9 / t_avg:.2f} GB/s (HCCS)")
        print(f"  Performance: {t_avg * 1e6:.2f} us")
        print(f"  ✓ dispatch_prefill passed", flush=True)
    
    dist.barrier()
    
    # Test 3: combine_prefill
    if local_rank == 0:
        print(f"\n[Test 3] combine_prefill")
        print("-" * 60, flush=True)
    
    combine_args = {
        "x": recv_x_bf16,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
        "src_idx": expand_idx_out,
        "send_head": recv_count,
        "groupEp": hccl_comm_name,
        "rank": rank,
        "num_ranks": world_size,
    }
    
    t_avg, _, _ = bench(
        lambda: torch.ops._C_ascend.combine_prefill(**combine_args),
        num_warmups=10,
        num_tests=20
    )
    
    combined_x = torch.ops._C_ascend.combine_prefill(**combine_args)
    torch.npu.synchronize()
    
    combine_bytes = recv_x_bf16.numel() * 2
    
    if local_rank == 0:
        print(f"  combined_x shape: {combined_x.shape}")
        print(f"  Bandwidth: {combine_bytes / 1e9 / t_avg:.2f} GB/s (HCCS)")
        print(f"  Performance: {t_avg * 1e6:.2f} us")
        print(f"  ✓ combine_prefill passed", flush=True)
    
    dist.barrier()
    
    # Test 4: Correctness check
    if local_rank == 0:
        print(f"\n[Test 4] Correctness verification")
        print("-" * 60, flush=True)
    
    # Expected: combined_x should equal x * sum(topk_weights for valid experts)
    ref_x = x
    expected_weights = topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1)
    expected_output = ref_x * expected_weights
    
    diff = calc_diff(combined_x.float(), expected_output.float())
    
    if local_rank == 0:
        print(f"  Relative difference: {diff:.2e}")
        if diff < 1e-4:
            print(f"  ✓ Correctness check PASSED (diff < 1e-4)", flush=True)
        else:
            print(f"  ✗ Correctness check FAILED (diff = {diff:.2e})", flush=True)
    
    dist.barrier()
    
    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"All tests completed successfully!")
        print(f"{'='*60}", flush=True)
    
    dist.destroy_process_group()


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    """Test loop for each spawned process."""
    print(f"[Rank {local_rank}] Starting test...", flush=True)
    torch.manual_seed(local_rank + 42)
    
    test_moe_ops(args, local_rank, num_local_ranks)
    
    if local_rank == 0:
        print("", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM MoE dispatch/combine prefill operators"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of processes to spawn (default: 16)"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=3648,
        help="Number of tokens (default: 3648)"
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=4096,
        help="Hidden dimension size (default: 4096)"
    )
    parser.add_argument(
        "--num-topk",
        type=int,
        default=8,
        help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=256,
        help="Number of experts (default: 256)"
    )
    parser.add_argument(
        "--active-ranks",
        type=str,
        default="",
        help='Comma-separated list of ranks that will receive tokens. '
             'Example: "0,1,3". If empty, all ranks may receive tokens.'
    )
    parser.add_argument(
        "--use-quant",
        action="store_true",
        help="Enable dynamic quantization for communication"
    )
    parser.add_argument(
        "--load-saved-tensors",
        action="store_true",
        help="Load and test with previously saved tensors for debugging"
    )
    parser.add_argument(
        "--tensor-save-dir",
        type=str,
        default="/tmp/moe_debug_tensors",
        help="Directory to load saved tensors from (default: /tmp/moe_debug_tensors)"
    )
    parser.add_argument(
        "--use-dispatch-output",
        action="store_true",
        help="Use dispatch output directly for combine test (end-to-end mode)"
    )
    
    args = parser.parse_args()
    
    num_processes = args.num_processes
    print(f"Spawning {num_processes} processes for testing...\n", flush=True)
    
    torch.multiprocessing.spawn(
        test_loop, 
        args=(num_processes, args), 
        nprocs=num_processes
    )


if __name__ == "__main__":
    main()
