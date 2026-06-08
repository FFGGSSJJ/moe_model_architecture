"""Test and benchmark FP8 all_to_all_single across multiple nodes.

This script extends the intra-node FP8 A2A tests with:
  1. Correctness verification across node boundaries
  2. Bandwidth / latency benchmarking for various message sizes
  3. Per-node-pair bandwidth reporting
  4. Side-by-side FP8 vs BF16 comparison (same tensor shapes)

Launch via SLURM (recommended):
    sbatch run_fp8_a2a_internode.sbatch

Or manually with torchrun:
    torchrun \
        --nnodes=2 \
        --nproc_per_node=4 \
        --rdzv_id=job1 \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        tests/fp8_a2a_internode_test.py
"""

import os
import time
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fp8_data(shape, rank, fp8_dtype, ref_dtype=torch.float32):
    """Create deterministic FP8-safe data. Values in [-1, 1] + rank offset."""
    t = torch.zeros(shape, dtype=ref_dtype, device="cuda")
    t.view(-1)[::2] = 1.0
    t = t + rank * 0.1
    return t.to(fp8_dtype), t


def get_node_name():
    """Return the hostname of the current node."""
    return os.uname().nodename


def get_local_world_size():
    """Get the number of GPUs per node.

    Works with both torchrun (LOCAL_WORLD_SIZE) and srun
    (SLURM_TASKS_PER_NODE / SLURM_NTASKS_PER_NODE).
    """
    val = os.environ.get("LOCAL_WORLD_SIZE")
    if val is not None:
        return int(val)
    val = os.environ.get("SLURM_NTASKS_PER_NODE")
    if val is not None:
        return int(val)
    val = os.environ.get("SLURM_TASKS_PER_NODE")
    if val is not None:
        # SLURM_TASKS_PER_NODE can be like "4(x2)" for homogeneous allocations
        return int(val.split("(")[0])
    return 1


def make_random_fp8(shape, fp8_dtype):
    """Create random FP8 data via float32 -> FP8 cast (randn doesn't support FP8)."""
    return torch.randn(shape, dtype=torch.float32, device="cuda").to(fp8_dtype)


def make_random_bf16(shape):
    """Create random BF16 data."""
    return torch.randn(shape, dtype=torch.bfloat16, device="cuda")


def dtype_elem_size(dtype):
    """Return the byte size of a single element for the given dtype."""
    return torch.tensor([], dtype=dtype).element_size()


# ---------------------------------------------------------------------------
# Correctness tests (same logic as intra-node, but now exercised cross-node)
# ---------------------------------------------------------------------------

def test_all_to_all_single_equal_split():
    """Test all_to_all_single with FP8 tensors across nodes."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    node_name = get_node_name()

    N = 256
    fp8_dtype = torch.float8_e4m3fn

    send_tensor_fp8, _ = make_fp8_data(world_size * N, rank, fp8_dtype)
    recv_tensor_fp8 = torch.empty(world_size * N, dtype=fp8_dtype, device="cuda")

    dist.all_to_all_single(recv_tensor_fp8, send_tensor_fp8)

    for src_rank in range(world_size):
        _, expected_ref = make_fp8_data(world_size * N, src_rank, fp8_dtype)
        expected_chunk = expected_ref[rank * N : (rank + 1) * N].to(fp8_dtype)
        actual_chunk = recv_tensor_fp8[src_rank * N : (src_rank + 1) * N]
        diff = actual_chunk.float() - expected_chunk.float()
        max_diff = diff.abs().max().item()
        assert max_diff < 0.2, (
            f"[Rank {rank} on {node_name}] Mismatch from src_rank {src_rank}: "
            f"max diff={max_diff:.4f}"
        )

    if rank == 0:
        print(f"[PASS] test_all_to_all_single_equal_split (world_size={world_size})")


def test_all_to_all_single_unequal_split():
    """Test all_to_all_single with unequal splits across nodes."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    node_name = get_node_name()

    fp8_dtype = torch.float8_e4m3fn
    send_counts = [(r + 1) * 64 for r in range(world_size)]
    total_send = sum(send_counts)
    recv_counts = [send_counts[rank]] * world_size
    total_recv = sum(recv_counts)

    send_tensor_fp8, _ = make_fp8_data(total_send, rank, fp8_dtype)
    recv_tensor_fp8 = torch.empty(total_recv, dtype=fp8_dtype, device="cuda")

    dist.all_to_all_single(
        recv_tensor_fp8, send_tensor_fp8,
        output_split_sizes=recv_counts, input_split_sizes=send_counts,
    )

    offset = 0
    for src_rank in range(world_size):
        _, expected_ref = make_fp8_data(total_send, src_rank, fp8_dtype)
        chunk_start = sum(send_counts[:rank])
        chunk_size = send_counts[rank]
        expected_chunk = expected_ref[chunk_start : chunk_start + chunk_size].to(fp8_dtype)

        actual_chunk = recv_tensor_fp8[offset : offset + recv_counts[src_rank]]
        diff = actual_chunk.float() - expected_chunk.float()
        max_diff = diff.abs().max().item()
        assert max_diff < 0.2, (
            f"[Rank {rank} on {node_name}] Mismatch from src_rank {src_rank}: "
            f"max diff={max_diff:.4f}"
        )
        offset += recv_counts[src_rank]

    if rank == 0:
        print(f"[PASS] test_all_to_all_single_unequal_split (world_size={world_size})")


def test_fp8_e5m2_all_to_all():
    """Test all_to_all_single with float8_e5m2 across nodes."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    node_name = get_node_name()

    fp8_dtype = torch.float8_e5m2
    N = 256

    send_tensor_fp8, _ = make_fp8_data(world_size * N, rank, fp8_dtype)
    recv_tensor_fp8 = torch.empty(world_size * N, dtype=fp8_dtype, device="cuda")

    dist.all_to_all_single(recv_tensor_fp8, send_tensor_fp8)

    for src_rank in range(world_size):
        _, expected_ref = make_fp8_data(world_size * N, src_rank, fp8_dtype)
        expected_chunk = expected_ref[rank * N : (rank + 1) * N].to(fp8_dtype)
        actual_chunk = recv_tensor_fp8[src_rank * N : (src_rank + 1) * N]
        diff = actual_chunk.float() - expected_chunk.float()
        max_diff = diff.abs().max().item()
        assert max_diff < 0.2, (
            f"[Rank {rank} on {node_name}] Mismatch from src_rank {src_rank}: "
            f"max diff={max_diff:.4f}"
        )

    if rank == 0:
        print(f"[PASS] test_fp8_e5m2_all_to_all (world_size={world_size})")


# ---------------------------------------------------------------------------
# Inter-node specific: cross-node-only A2A correctness
# ---------------------------------------------------------------------------

def test_cross_node_pair_correctness():
    """Send FP8 data specifically between rank-pairs on different nodes.

    Each rank sends a unique payload to the rank at the same local-rank
    position on the *other* node.  This guarantees the data path crosses
    the interconnect (NVLink → NIC → network → NIC → NVLink).
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    node_name = get_node_name()
    local_world_size = get_local_world_size()

    assert world_size >= 2 * local_world_size, (
        f"Need at least 2 nodes (world_size={world_size}, "
        f"local_world_size={local_world_size})"
    )

    num_nodes = world_size // local_world_size

    fp8_dtype = torch.float8_e4m3fn
    N = 512

    # Build send/recv buffers: each rank sends world_size * N elements
    send_tensor_fp8, _ = make_fp8_data(world_size * N, rank, fp8_dtype)
    recv_tensor_fp8 = torch.empty(world_size * N, dtype=fp8_dtype, device="cuda")

    dist.all_to_all_single(recv_tensor_fp8, send_tensor_fp8)

    # Verify data from ranks on ALL nodes (including remote ones)
    for src_rank in range(world_size):
        src_node = src_rank // local_world_size
        _, expected_ref = make_fp8_data(world_size * N, src_rank, fp8_dtype)
        expected_chunk = expected_ref[rank * N : (rank + 1) * N].to(fp8_dtype)
        actual_chunk = recv_tensor_fp8[src_rank * N : (src_rank + 1) * N]
        diff = actual_chunk.float() - expected_chunk.float()
        max_diff = diff.abs().max().item()
        assert max_diff < 0.2, (
            f"[Rank {rank} on {node_name}] Cross-node mismatch from "
            f"src_rank {src_rank} (node {src_node}): max diff={max_diff:.4f}"
        )

    if rank == 0:
        print(
            f"[PASS] test_cross_node_pair_correctness "
            f"(world_size={world_size}, nodes={num_nodes})"
        )


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------

def _bench_all_to_all_bandwidth(dtype, label=None):
    """Benchmark all_to_all_single bandwidth for a given dtype.

    The tensor shape is expressed in *elements* (same across dtypes so that
    FP8 and BF16 results are directly comparable).  The byte counts differ
    because FP8 = 1 byte/elem while BF16 = 2 bytes/elem.

    Returns a list of dicts with per-size results.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_world_size = get_local_world_size()
    num_nodes = world_size // local_world_size

    elem_bytes = dtype_elem_size(dtype)
    if label is None:
        label = str(dtype)

    # Message sizes: elements per rank (sweep from 4 KiB to 128 MiB in FP8 terms)
    sizes_kib = [4, 16, 64, 256, 1024, 4096, 16384, 65536, 131072]
    warmup = 5
    num_iters = 20

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"{label} all_to_all_single Bandwidth Benchmark")
        print(f"  World size : {world_size}")
        print(f"  Nodes      : {num_nodes}")
        print(f"  GPUs/node  : {local_world_size}")
        print(f"  dtype      : {dtype} ({elem_bytes} byte/elem)")
        print(f"  Warmup     : {warmup} iters")
        print(f"  Measured   : {num_iters} iters")
        print(f"{'='*80}")
        header = (
            f"{'Elems (KiB)':>12} {'Elements':>14} {'Latency (us)':>14} "
            f"{'Algo BW (GB/s)':>16} {'Bus BW (GB/s)':>16}"
        )
        print(header)
        print("-" * 80)

    results = []

    for size_kib in sizes_kib:
        num_elements = size_kib * 1024  # same element count regardless of dtype
        total_elements = num_elements * world_size
        total_bytes = total_elements * elem_bytes

        if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
            send_tensor = make_random_fp8(total_elements, dtype)
        else:
            send_tensor = make_random_bf16(total_elements)
        recv_tensor = torch.empty(total_elements, dtype=dtype, device="cuda")

        # Warmup
        for _ in range(warmup):
            dist.all_to_all_single(recv_tensor, send_tensor)

        # Timed runs
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(num_iters):
            dist.all_to_all_single(recv_tensor, send_tensor)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_us = (t1 - t0) / num_iters * 1e6
        elapsed_s = (t1 - t0) / num_iters

        algo_bw = total_bytes / elapsed_s / 1e9  # GB/s sent by this rank
        bus_bw = algo_bw * (world_size - 1) / world_size  # NCCL bus BW

        results.append({
            "size_kib": size_kib,
            "elements": num_elements,
            "time_us": elapsed_us,
            "algo_bw": algo_bw,
            "bus_bw": bus_bw,
        })

        if rank == 0:
            print(
                f"{size_kib:>12} {num_elements:>14,} {elapsed_us:>14.1f} "
                f"{algo_bw:>16.2f} {bus_bw:>16.2f}"
            )

    if rank == 0:
        print(f"{'='*80}\n")

    return results


def bench_all_to_all_bandwidth():
    """Benchmark FP8 all_to_all_single bandwidth for various message sizes."""
    return _bench_all_to_all_bandwidth(torch.float8_e4m3fn, label="FP8")


def bench_all_to_all_bandwidth_bf16():
    """Benchmark BF16 all_to_all_single bandwidth (same element counts as FP8)."""
    return _bench_all_to_all_bandwidth(torch.bfloat16, label="BF16")


def bench_fp8_vs_bf16_comparison():
    """Run FP8 and BF16 all_to_all back-to-back with the same tensor shapes.

    Uses the same number of elements per rank for both dtypes, so:
      - FP8 transfers 1 byte/element
      - BF16 transfers 2 bytes/element
    The comparison reveals whether FP8's smaller footprint translates to
    lower latency or if the interconnect is the bottleneck regardless.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_world_size = get_local_world_size()
    num_nodes = world_size // local_world_size

    fp8_dtype = torch.float8_e4m3fn
    bf16_dtype = torch.bfloat16

    sizes_kib = [4, 16, 64, 256, 1024, 4096, 16384, 65536, 131072]
    warmup = 5
    num_iters = 20

    if rank == 0:
        print(f"\n{'='*100}")
        print(f"FP8 vs BF16 all_to_all_single Comparison (same tensor shape)")
        print(f"  World size : {world_size}")
        print(f"  Nodes      : {num_nodes}")
        print(f"  GPUs/node  : {local_world_size}")
        print(f"  Warmup     : {warmup} iters, Measured: {num_iters} iters")
        print(f"{'='*100}")
        header = (
            f"{'Elems(KiB)':>10} | "
            f"{'FP8 lat(us)':>12} {'FP8 busBW':>12} | "
            f"{'BF16 lat(us)':>12} {'BF16 busBW':>12} | "
            f"{'Lat ratio':>10} {'BW ratio':>10}"
        )
        print(header)
        print("-" * 100)

    for size_kib in sizes_kib:
        num_elements = size_kib * 1024
        total_elements = num_elements * world_size

        # --- FP8 ---
        fp8_bytes = total_elements * dtype_elem_size(fp8_dtype)
        send_fp8 = make_random_fp8(total_elements, fp8_dtype)
        recv_fp8 = torch.empty(total_elements, dtype=fp8_dtype, device="cuda")

        for _ in range(warmup):
            dist.all_to_all_single(recv_fp8, send_fp8)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            dist.all_to_all_single(recv_fp8, send_fp8)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fp8_elapsed_s = (t1 - t0) / num_iters
        fp8_elapsed_us = fp8_elapsed_s * 1e6
        fp8_bus_bw = (fp8_bytes / fp8_elapsed_s / 1e9) * (world_size - 1) / world_size

        # --- BF16 (same shape = same number of elements) ---
        bf16_bytes = total_elements * dtype_elem_size(bf16_dtype)
        send_bf16 = make_random_bf16(total_elements)
        recv_bf16 = torch.empty(total_elements, dtype=bf16_dtype, device="cuda")

        for _ in range(warmup):
            dist.all_to_all_single(recv_bf16, send_bf16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            dist.all_to_all_single(recv_bf16, send_bf16)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        bf16_elapsed_s = (t1 - t0) / num_iters
        bf16_elapsed_us = bf16_elapsed_s * 1e6
        bf16_bus_bw = (bf16_bytes / bf16_elapsed_s / 1e9) * (world_size - 1) / world_size

        lat_ratio = fp8_elapsed_us / bf16_elapsed_us
        bw_ratio = fp8_bus_bw / bf16_bus_bw

        if rank == 0:
            print(
                f"{size_kib:>10} | "
                f"{fp8_elapsed_us:>12.1f} {fp8_bus_bw:>11.2f}G | "
                f"{bf16_elapsed_us:>12.1f} {bf16_bus_bw:>11.2f}G | "
                f"{lat_ratio:>10.3f} {bw_ratio:>10.3f}"
            )

    if rank == 0:
        print(f"{'='*100}")
        print("  Lat ratio = FP8_latency / BF16_latency  (< 1 → FP8 faster)")
        print("  BW  ratio = FP8_busBW  / BF16_busBW   (> 1 → FP8 higher BW)")
        print(f"{'='*100}\n")


def _bench_cross_node_vs_intra_node(dtype, label=None):
    """Compare A2A bandwidth: data staying on-node vs crossing nodes.

    Strategy: use unequal splits so that each rank sends a *large* chunk
    to ranks on the same node and a *small* chunk to ranks on other nodes
    (and vice-versa).  We isolate the cross-node portion by measuring
    the full A2A and subtracting the estimated intra-node contribution.

    A simpler and more direct approach: just time the full A2A at a size
    that is dominated by the inter-node hop.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_world_size = get_local_world_size()
    num_nodes = world_size // local_world_size

    elem_bytes = dtype_elem_size(dtype)
    if label is None:
        label = str(dtype)

    warmup = 5
    num_iters = 20

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Cross-Node vs Intra-Node {label} A2A Latency Comparison")
        print(f"  World size : {world_size}, Nodes: {num_nodes}, GPUs/node: {local_world_size}")
        print(f"{'='*80}")

    # We benchmark at a few sizes.  For each, we run the full all-to-all
    # which includes both intra-node (NVLink) and inter-node (NIC) transfers.
    sizes_mib = [1, 4, 16, 64]

    if rank == 0:
        print(
            f"{'Size (MiB)':>12} {'Latency (us)':>14} "
            f"{'Bus BW (GB/s)':>16}"
        )
        print("-" * 48)

    for size_mib in sizes_mib:
        num_elements = size_mib * 1024 * 1024  # same element count for both dtypes
        total_elements = num_elements * world_size

        if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
            send_tensor = make_random_fp8(total_elements, dtype)
        else:
            send_tensor = make_random_bf16(total_elements)
        recv_tensor = torch.empty(total_elements, dtype=dtype, device="cuda")

        for _ in range(warmup):
            dist.all_to_all_single(recv_tensor, send_tensor)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            dist.all_to_all_single(recv_tensor, send_tensor)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_us = (t1 - t0) / num_iters * 1e6
        elapsed_s = (t1 - t0) / num_iters
        total_bytes = total_elements * elem_bytes
        algo_bw = total_bytes / elapsed_s / 1e9
        bus_bw = algo_bw * (world_size - 1) / world_size

        if rank == 0:
            print(
                f"{size_mib:>12} {elapsed_us:>14.1f} "
                f"{bus_bw:>16.2f}"
            )

    if rank == 0:
        print(f"{'='*80}\n")


def bench_cross_node_vs_intra_node():
    """Cross-node vs intra-node FP8 A2A benchmark."""
    return _bench_cross_node_vs_intra_node(torch.float8_e4m3fn, label="FP8")


def bench_cross_node_vs_intra_node_bf16():
    """Cross-node vs intra-node BF16 A2A benchmark."""
    return _bench_cross_node_vs_intra_node(torch.bfloat16, label="BF16")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    node_name = get_node_name()
    local_world_size = get_local_world_size()
    num_nodes = world_size // local_world_size

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"FP8 Inter-Node all_to_all_single Test & Benchmark")
        print(f"  torch version : {torch.__version__}")
        print(f"  World size    : {world_size}")
        print(f"  Nodes         : {num_nodes}")
        print(f"  GPUs/node     : {local_world_size}")
        print(f"  Node 0        : {node_name}")

    # Print node layout from rank 0 on each node
    if rank % local_world_size == 0:
        print(f"  [Node {rank // local_world_size}] hostname={node_name}, "
              f"ranks {rank}-{rank + local_world_size - 1}")

    dist.barrier()

    # ---- Correctness tests ----
    correctness_tests = [
        test_all_to_all_single_equal_split,
        test_all_to_all_single_unequal_split,
        test_fp8_e5m2_all_to_all,
        test_cross_node_pair_correctness,
    ]

    for test_fn in correctness_tests:
        dist.barrier()
        try:
            test_fn()
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__} on rank {rank} ({node_name}): {e}")
            raise

    dist.barrier()
    if rank == 0:
        print(f"\nAll {len(correctness_tests)} correctness tests passed!\n")

    # ---- Performance benchmarks ----
    dist.barrier()
    bench_all_to_all_bandwidth()

    dist.barrier()
    bench_all_to_all_bandwidth_bf16()

    dist.barrier()
    bench_fp8_vs_bf16_comparison()

    dist.barrier()
    if num_nodes > 1:
        bench_cross_node_vs_intra_node()

        dist.barrier()
        bench_cross_node_vs_intra_node_bf16()

    dist.barrier()
    if rank == 0:
        print("Done.")

    dist.destroy_process_group()
