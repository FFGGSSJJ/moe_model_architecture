"""Test FP8 tensor support for torch.distributed.all_to_all_single.

Launch with:
    torchrun --nproc_per_node=2 grouped_gemm/fp8_a2a_test.py
    torchrun --nproc_per_node=4 grouped_gemm/fp8_a2a_test.py
"""

import torch
import torch.distributed as dist


def make_fp8_data(shape, rank, fp8_dtype, ref_dtype=torch.float32):
    """Create deterministic FP8-safe data. Values in [-1, 1] + rank offset."""
    t = torch.zeros(shape, dtype=ref_dtype, device="cuda")
    t.view(-1)[::2] = 1.0
    t = t + rank * 0.1
    return t.to(fp8_dtype), t


def test_all_to_all_single_equal_split():
    """Test all_to_all_single with FP8 tensors where each rank sends an equal-sized chunk."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    N = 256
    fp8_dtype = torch.float8_e4m3fn

    send_tensor_fp8, _ = make_fp8_data(world_size * N, rank, fp8_dtype)
    recv_tensor_fp8 = torch.empty(world_size * N, dtype=fp8_dtype, device="cuda")

    dist.all_to_all_single(recv_tensor_fp8, send_tensor_fp8)

    # Rank r receives chunk r from each src_rank
    for src_rank in range(world_size):
        _, expected_ref = make_fp8_data(world_size * N, src_rank, fp8_dtype)
        expected_chunk = expected_ref[rank * N : (rank + 1) * N].to(fp8_dtype)
        actual_chunk = recv_tensor_fp8[src_rank * N : (src_rank + 1) * N]
        diff = actual_chunk.float() - expected_chunk.float()
        max_diff = diff.abs().max().item()
        assert max_diff < 0.2, (
            f"[Rank {rank}] Mismatch from src_rank {src_rank}: max diff={max_diff:.4f}"
        )

    if rank == 0:
        print(f"[PASS] test_all_to_all_single_equal_split (world_size={world_size})")


def test_all_to_all_single_unequal_split():
    """Test all_to_all_single with FP8 tensors using split/sizes arguments.

    Rank i splits its send tensor into chunks of sizes send_counts,
    sending chunk j to rank j.  Rank i receives recv_counts[j] elements
    from rank j, where recv_counts[j] = send_counts_of_rank_j[i].
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    fp8_dtype = torch.float8_e4m3fn

    # send_counts[r] = how many elements rank r sends to each other rank
    send_counts = [(r + 1) * 64 for r in range(world_size)]
    total_send = sum(send_counts)

    # recv_counts[j] = how many elements we receive from rank j
    # = send_counts_of_rank_j[our_rank] = send_counts[rank]
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
        # src_rank splits with send_counts; chunk for our rank starts at sum(send_counts[:rank])
        chunk_start = sum(send_counts[:rank])
        chunk_size = send_counts[rank]
        expected_chunk = expected_ref[chunk_start:chunk_start + chunk_size].to(fp8_dtype)

        actual_chunk = recv_tensor_fp8[offset : offset + recv_counts[src_rank]]
        diff = actual_chunk.float() - expected_chunk.float()
        max_diff = diff.abs().max().item()
        assert max_diff < 0.2, (
            f"[Rank {rank}] Mismatch from src_rank {src_rank}: max diff={max_diff:.4f}"
        )
        offset += recv_counts[src_rank]

    if rank == 0:
        print(f"[PASS] test_all_to_all_single_unequal_split (world_size={world_size})")


def test_all_to_all_2d_tensor():
    """Test all_to_all_single with 2D FP8 tensors."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    fp8_dtype = torch.float8_e4m3fn
    rows, cols = world_size, 128

    send_tensor_fp8, _ = make_fp8_data((rows, cols), rank, fp8_dtype)
    recv_tensor_fp8 = torch.empty(rows, cols, dtype=fp8_dtype, device="cuda")

    dist.all_to_all_single(recv_tensor_fp8, send_tensor_fp8)

    for src_rank in range(world_size):
        row = recv_tensor_fp8[src_rank : src_rank + 1, :]
        assert row.shape == (1, cols), f"[Rank {rank}] Wrong shape from src_rank {src_rank}"
        vals = row.float()
        assert vals.isfinite().all(), f"[Rank {rank}] Non-finite values from src_rank {src_rank}"
        # Values should be near src_rank * 0.1 or src_rank * 0.1 + 1.0
        expected_min = src_rank * 0.1 - 0.2
        expected_max = src_rank * 0.1 + 1.0 + 0.2
        assert vals.min().item() >= expected_min and vals.max().item() <= expected_max, (
            f"[Rank {rank}] Values out of expected range from src_rank {src_rank}: "
            f"[{vals.min().item():.3f}, {vals.max().item():.3f}]"
        )

    if rank == 0:
        print(f"[PASS] test_all_to_all_2d_tensor (world_size={world_size})")


def test_fp8_e5m2_all_to_all():
    """Test all_to_all_single with float8_e5m2 dtype."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

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
            f"[Rank {rank}] Mismatch from src_rank {src_rank}: max diff={max_diff:.4f}"
        )

    if rank == 0:
        print(f"[PASS] test_fp8_e5m2_all_to_all (world_size={world_size})")


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"Testing FP8 all_to_all_single with {world_size} GPUs")
        print(f"torch version: {torch.__version__}")

    tests = [
        test_all_to_all_single_equal_split,
        test_all_to_all_single_unequal_split,
        test_all_to_all_2d_tensor,
        test_fp8_e5m2_all_to_all,
    ]

    for test_fn in tests:
        dist.barrier()
        try:
            test_fn()
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__} on rank {rank}: {e}")
            raise

    dist.barrier()
    if rank == 0:
        print(f"\nAll {len(tests)} tests passed!")

    dist.destroy_process_group()
