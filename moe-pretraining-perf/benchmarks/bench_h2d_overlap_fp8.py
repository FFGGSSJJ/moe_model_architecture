import torch
import deep_gemm
import grouped_gemm

EDP = 8
NUM_LOCAL_LAYERS = 4
NUM_LOCAL_EXPERTS = 64

# Overlapping
CHUNK_SIZE = 16
HIDDEN_SIZE = 4096
MOE_FFN_HIDDEN_SIZE = 2048
TOKEN_PER_EXPERT = 1024

WARMUP = 5
ITERS = 50

FP8_MAX = 448.0


# ---------------------------------------------------------------------------
# FP8 quantization helpers (match DeepGEMM scale layouts)
# ---------------------------------------------------------------------------

def per_token_quant(x, gran_k=128):
    """Per-token-group FP8 quantization: one fp32 scale per (row, gran_k) group.
    Returns (x_fp8 [M, N], scales [M, N // gran_k]).
    """
    m, n = x.shape
    assert n % gran_k == 0
    num_groups = n // gran_k
    x_f = x.float()
    x_grouped = x_f.view(m, num_groups, gran_k)
    scales = x_grouped.abs().amax(dim=-1, keepdim=True) / FP8_MAX
    scales = scales.clamp(min=1e-12)
    x_fp8 = (x_grouped / scales).to(torch.float8_e4m3fn).view(m, n)
    return x_fp8, scales.squeeze(-1)


def per_block_quant(x, gran_k=128):
    """Per-block FP8 quantization: one fp32 scale per (gran_k, gran_k) block.
    Returns (x_fp8 [M, N], scales [M // gran_k, N // gran_k]).
    """
    m, n = x.shape
    assert m % gran_k == 0 and n % gran_k == 0
    bm, bn = m // gran_k, n // gran_k
    x_f = x.float()
    x_blocked = x_f.view(bm, gran_k, bn, gran_k)
    scales = x_blocked.abs().amax(dim=(1, 3), keepdim=True) / FP8_MAX
    scales = scales.clamp(min=1e-12)
    x_fp8 = (x_blocked / scales).to(torch.float8_e4m3fn).view(m, n)
    return x_fp8, scales.view(bm, bn)


# ---------------------------------------------------------------------------
# H2D benchmark
# ---------------------------------------------------------------------------

def bench_h2d():
    """Benchmark FP8 H2D async transfer of expert weights."""

    def _h2d(srcs, dsts):
        stream = torch.cuda.Stream()
        for _ in range(WARMUP):
            grouped_gemm.grouped_gemm.backend.batched_h2d_async(
                srcs, dsts, stream.cuda_stream,
            )
        stream.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(ITERS):
            grouped_gemm.grouped_gemm.backend.batched_h2d_async(
                srcs, dsts, stream.cuda_stream,
            )
        end.record(stream)
        torch.cuda.synchronize()
        return start.elapsed_time(end) / ITERS

    torch.manual_seed(0)
    num_tensors = CHUNK_SIZE

    # FC1: per-expert weight (N, K) = (ffn*2, hidden) in FP8
    srcs, dsts = [], []
    for _ in range(num_tensors):
        srcs.append(
            torch.empty(
                MOE_FFN_HIDDEN_SIZE * 2, HIDDEN_SIZE,
                dtype=torch.float8_e4m3fn, pin_memory=True, device="cpu",
            )
        )
        dsts.append(
            torch.empty(
                MOE_FFN_HIDDEN_SIZE * 2, HIDDEN_SIZE,
                device="cuda:0", dtype=torch.float8_e4m3fn,
            )
        )
    total_bytes = sum(s.numel() * s.element_size() for s in srcs)
    fc1_elapsed_ms = _h2d(srcs, dsts)
    fc1_bw = total_bytes / (fc1_elapsed_ms * 1e-3) / 1e9

    # FC2: per-expert weight (N, K) = (hidden, ffn) in FP8
    srcs, dsts = [], []
    for _ in range(num_tensors):
        srcs.append(
            torch.empty(
                HIDDEN_SIZE, MOE_FFN_HIDDEN_SIZE,
                dtype=torch.float8_e4m3fn, pin_memory=True, device="cpu",
            )
        )
        dsts.append(
            torch.empty(
                HIDDEN_SIZE, MOE_FFN_HIDDEN_SIZE,
                device="cuda:0", dtype=torch.float8_e4m3fn,
            )
        )
    total_bytes = sum(s.numel() * s.element_size() for s in srcs)
    fc2_elapsed_ms = _h2d(srcs, dsts)
    fc2_bw = total_bytes / (fc2_elapsed_ms * 1e-3) / 1e9

    print(f"[FP8 H2D] {num_tensors} experts")
    print(f"  FC1 shape per expert: ({MOE_FFN_HIDDEN_SIZE*2}, {HIDDEN_SIZE}), dtype: float8_e4m3fn")
    print(f"  FC1 Latency: {fc1_elapsed_ms:.3f} ms | BW: {fc1_bw:.2f} GB/s | MBU: {fc1_bw / 450 * 100:.2f}%")
    print(f"  FC2 shape per expert: ({HIDDEN_SIZE}, {MOE_FFN_HIDDEN_SIZE}), dtype: float8_e4m3fn")
    print(f"  FC2 Latency: {fc2_elapsed_ms:.3f} ms | BW: {fc2_bw:.2f} GB/s | MBU: {fc2_bw / 450 * 100:.2f}%")
    return fc1_elapsed_ms, fc2_elapsed_ms


# ---------------------------------------------------------------------------
# FP8 Grouped GEMM benchmark (DeepGEMM m-grouped contiguous)
# ---------------------------------------------------------------------------

def bench_fp8_groupgemm():
    """Benchmark FP8 Grouped GEMM with DeepGEMM (m-grouped contiguous NT layout)."""

    def _fp8_gemm(a, sfa, b, sfb, tokens_psum):
        output = torch.empty(
            a.shape[0], b.shape[1],
            dtype=torch.bfloat16, device=a.device,
        )
        for _ in range(WARMUP):
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (a, sfa), (b, sfb), output, tokens_psum,
                recipe_a=(1, 128), recipe_b=(128, 128),
                disable_ue8m0_cast=True, use_psum_layout=True,
            )
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (a, sfa), (b, sfb), output, tokens_psum,
                recipe_a=(1, 128), recipe_b=(128, 128),
                disable_ue8m0_cast=True, use_psum_layout=True,
            )
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / ITERS

    torch.manual_seed(0)
    num_experts = CHUNK_SIZE
    tokens_per_expert = TOKEN_PER_EXPERT
    hidden = HIDDEN_SIZE
    ffn = MOE_FFN_HIDDEN_SIZE
    total_tokens = num_experts * tokens_per_expert

    # Cumulative sum of tokens per expert (DeepGEMM contiguous psum layout)
    tokens_psum = torch.arange(
        tokens_per_expert, total_tokens + 1, tokens_per_expert,
        dtype=torch.int32, device="cuda:0",
    )

    # --- FC1: A[M, K] @ B[G, N, K].T -> C[M, N] ---
    # A: [total_tokens, hidden], per-token quant -> sfa [M, K//128]
    a_bf16 = torch.randn(total_tokens, hidden, device="cuda:0", dtype=torch.bfloat16)
    fp8_a, a_scales = per_token_quant(a_bf16)

    # B: [num_experts, ffn*2, hidden], per-block quant -> sfb [G, N//128, K//128]
    b_bf16 = torch.randn(num_experts, ffn * 2, hidden, device="cuda:0", dtype=torch.bfloat16)
    fp8_b_list, b_scales_list = [], []
    for i in range(num_experts):
        w, sf = per_block_quant(b_bf16[i])
        fp8_b_list.append(w)
        b_scales_list.append(sf)
    fp8_b = torch.stack(fp8_b_list)
    b_scales = torch.stack(b_scales_list)

    fc1_elapsed_ms = _fp8_gemm(fp8_a, a_scales, fp8_b, b_scales, tokens_psum)
    fc1_flops = num_experts * 2.0 * tokens_per_expert * hidden * ffn * 2
    fc1_tflops = fc1_flops / (fc1_elapsed_ms * 1e-3) / 1e12

    # --- FC2: A[M, K] @ B[G, N, K].T -> C[M, N] ---
    a_bf16 = torch.randn(total_tokens, ffn, device="cuda:0", dtype=torch.bfloat16)
    fp8_a, a_scales = per_token_quant(a_bf16)

    b_bf16 = torch.randn(num_experts, hidden, ffn, device="cuda:0", dtype=torch.bfloat16)
    fp8_b_list, b_scales_list = [], []
    for i in range(num_experts):
        w, sf = per_block_quant(b_bf16[i])
        fp8_b_list.append(w)
        b_scales_list.append(sf)
    fp8_b = torch.stack(fp8_b_list)
    b_scales = torch.stack(b_scales_list)

    fc2_elapsed_ms = _fp8_gemm(fp8_a, a_scales, fp8_b, b_scales, tokens_psum)
    fc2_flops = num_experts * 2.0 * tokens_per_expert * ffn * hidden
    fc2_tflops = fc2_flops / (fc2_elapsed_ms * 1e-3) / 1e12

    print(f"[FP8 GroupedGEMM FWD] {num_experts} experts, {tokens_per_expert} tokens/expert")
    print(f"  FC1: A({total_tokens}, {hidden}) @ B({num_experts}, {ffn*2}, {hidden}).T")
    print(f"  FC1 Latency: {fc1_elapsed_ms:.3f} ms | TFLOPS: {fc1_tflops:.2f} | MFU: {fc1_tflops / 989 * 100:.2f}%")
    print(f"  FC2: A({total_tokens}, {ffn}) @ B({num_experts}, {hidden}, {ffn}).T")
    print(f"  FC2 Latency: {fc2_elapsed_ms:.3f} ms | TFLOPS: {fc2_tflops:.2f} | MFU: {fc2_tflops / 989 * 100:.2f}%")
    return fc1_elapsed_ms, fc2_elapsed_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fc1_h2d_latency, fc2_h2d_latency = bench_h2d()
    fc1_gg_latency, fc2_gg_latency = bench_fp8_groupgemm()

    fc1_gap_ms = fc1_h2d_latency - fc1_gg_latency
    fc2_gap_ms = fc2_h2d_latency - fc2_gg_latency
    fc1_overlap_efficiency = fc1_gg_latency / fc1_h2d_latency
    fc2_overlap_efficiency = fc2_gg_latency / fc2_h2d_latency

    print(f"\nExposed H2D Latency: FC1={fc1_gap_ms:.3f} ms | FC2={fc2_gap_ms:.3f} ms")
    print(f"Overlap efficiency: FC1={fc1_overlap_efficiency*100:.2f}% | FC2={fc2_overlap_efficiency*100:.2f}%")
