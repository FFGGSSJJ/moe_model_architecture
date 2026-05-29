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

    print(f"\n{'─'*66}")
    print(f"  FP8 H2D Async Transfer · {num_tensors} experts")
    print(f"{'─'*66}")
    print(f"  FC1  ({MOE_FFN_HIDDEN_SIZE*2}, {HIDDEN_SIZE}) fp8e4")
    print(f"       Latency {fc1_elapsed_ms:.3f} ms · BW {fc1_bw:.2f} GB/s · MBU {fc1_bw / 450 * 100:.2f}%")
    print(f"  FC2  ({HIDDEN_SIZE}, {MOE_FFN_HIDDEN_SIZE}) fp8e4")
    print(f"       Latency {fc2_elapsed_ms:.3f} ms · BW {fc2_bw:.2f} GB/s · MBU {fc2_bw / 450 * 100:.2f}%")
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

    print(f"\n{'─'*66}")
    print(f"  FP8 GroupedGEMM FWD · {num_experts} experts · {tokens_per_expert} tok/exp")
    print(f"{'─'*66}")
    print(f"  FC1  A({total_tokens}, {hidden}) @ B({num_experts}, {ffn*2}, {hidden}).T")
    print(f"       Latency {fc1_elapsed_ms:.3f} ms · {fc1_tflops:.2f} TFLOPS · MFU {fc1_tflops / 989 * 100:.2f}%")
    print(f"  FC2  A({total_tokens}, {ffn}) @ B({num_experts}, {hidden}, {ffn}).T")
    print(f"       Latency {fc2_elapsed_ms:.3f} ms · {fc2_tflops:.2f} TFLOPS · MFU {fc2_tflops / 989 * 100:.2f}%")
    return fc1_elapsed_ms, fc2_elapsed_ms


# ---------------------------------------------------------------------------
# FP8 Grouped GEMM backward benchmark — grad_a: ds = grad_y @ w2.T
# ---------------------------------------------------------------------------

def bench_fp8_groupgemm_bwd_grad_a():
    """Benchmark FP8 Grouped GEMM for backward grad_a computation.

    grad_s [M, ffn] = grad_y [M, hidden] @ w2.T [G, ffn, hidden]
      — w2.T is the transposed second-layer weight, per-block quantised.
      — H2D transfer size is identical to forward FC2 (same element count).
    """

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

    tokens_psum = torch.arange(
        tokens_per_expert, total_tokens + 1, tokens_per_expert,
        dtype=torch.int32, device="cuda:0",
    )

    # A = grad_y: [total_tokens, hidden], per-token quant
    grad_y_bf16 = torch.randn(total_tokens, hidden, device="cuda:0", dtype=torch.bfloat16)
    fp8_grad_y, grad_y_scales = per_token_quant(grad_y_bf16)

    # B = w2 transposed: [num_experts, ffn, hidden], per-block quant
    w2_t_bf16 = torch.randn(num_experts, ffn, hidden, device="cuda:0", dtype=torch.bfloat16)
    fp8_w2_t_list, w2_t_scales_list = [], []
    for i in range(num_experts):
        w, sf = per_block_quant(w2_t_bf16[i])
        fp8_w2_t_list.append(w)
        w2_t_scales_list.append(sf)
    fp8_w2_t = torch.stack(fp8_w2_t_list)
    w2_t_scales = torch.stack(w2_t_scales_list)

    elapsed_ms = _fp8_gemm(fp8_grad_y, grad_y_scales, fp8_w2_t, w2_t_scales, tokens_psum)
    flops = num_experts * 2.0 * tokens_per_expert * hidden * ffn
    tflops = flops / (elapsed_ms * 1e-3) / 1e12

    print(f"\n{'─'*66}")
    print(f"  FP8 GroupedGEMM BWD · grad_a · {num_experts} experts · {tokens_per_expert} tok/exp")
    print(f"{'─'*66}")
    print(f"  grad_s = grad_y({total_tokens}, {hidden}) @ w2.T({num_experts}, {ffn}, {hidden})")
    print(f"       Latency {elapsed_ms:.3f} ms · {tflops:.2f} TFLOPS · MFU {tflops / 989 * 100:.2f}%")
    return elapsed_ms


# ---------------------------------------------------------------------------
# FP8 Grouped GEMM backward benchmark — grad_x: dx = grad_a @ w1.T
# ---------------------------------------------------------------------------

def bench_fp8_groupgemm_bwd_grad_x():
    """Benchmark FP8 Grouped GEMM for backward grad_x computation.

    dx [M, hidden] = grad_a [M, ffn*2] @ w1.T [G, hidden, ffn*2]
      — w1.T is the transposed first-layer weight, per-block quantised.
      — H2D transfer size is identical to forward FC1 (same element count).
    """

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

    tokens_psum = torch.arange(
        tokens_per_expert, total_tokens + 1, tokens_per_expert,
        dtype=torch.int32, device="cuda:0",
    )

    # A = grad_a: [total_tokens, ffn*2], per-token quant
    grad_a_bf16 = torch.randn(total_tokens, ffn * 2, device="cuda:0", dtype=torch.bfloat16)
    fp8_grad_a, grad_a_scales = per_token_quant(grad_a_bf16)

    # B = w1 transposed: [num_experts, hidden, ffn*2], per-block quant
    w1_t_bf16 = torch.randn(num_experts, hidden, ffn * 2, device="cuda:0", dtype=torch.bfloat16)
    fp8_w1_t_list, w1_t_scales_list = [], []
    for i in range(num_experts):
        w, sf = per_block_quant(w1_t_bf16[i])
        fp8_w1_t_list.append(w)
        w1_t_scales_list.append(sf)
    fp8_w1_t = torch.stack(fp8_w1_t_list)
    w1_t_scales = torch.stack(w1_t_scales_list)

    elapsed_ms = _fp8_gemm(fp8_grad_a, grad_a_scales, fp8_w1_t, w1_t_scales, tokens_psum)
    flops = num_experts * 2.0 * tokens_per_expert * ffn * 2 * hidden
    tflops = flops / (elapsed_ms * 1e-3) / 1e12

    print(f"\n{'─'*66}")
    print(f"  FP8 GroupedGEMM BWD · grad_x · {num_experts} experts · {tokens_per_expert} tok/exp")
    print(f"{'─'*66}")
    print(f"  dx = grad_a({total_tokens}, {ffn*2}) @ w1.T({num_experts}, {hidden}, {ffn*2})")
    print(f"       Latency {elapsed_ms:.3f} ms · {tflops:.2f} TFLOPS · MFU {tflops / 989 * 100:.2f}%")
    return elapsed_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Run benchmarks ──
    fc1_h2d, fc2_h2d = bench_h2d()
    fc1_fwd, fc2_fwd = bench_fp8_groupgemm()
    bwd_grad_a = bench_fp8_groupgemm_bwd_grad_a()
    bwd_grad_x = bench_fp8_groupgemm_bwd_grad_x()

    # ── Summary table ──
    # grad_a uses w2.T (same byte count as FC2 for H2D).
    # grad_x uses w1.T (same byte count as FC1 for H2D).
    rows = [
        ("FC1 (fwd)",    fc1_h2d, fc1_fwd),
        ("FC2 (fwd)",    fc2_h2d, fc2_fwd),
        ("grad_a (bwd)", fc2_h2d, bwd_grad_a),
        ("grad_x (bwd)", fc1_h2d, bwd_grad_x),
    ]

    print(f"\n{'━'*70}")
    print(f"  Overlap Efficiency Summary")
    print(f"  chunk={CHUNK_SIZE}  hidden={HIDDEN_SIZE}  ffn={MOE_FFN_HIDDEN_SIZE}"
          f"  tok/expert={TOKEN_PER_EXPERT}")
    print(f"{'━'*70}")
    print(f"  {'Layer':<15}{'H2D (ms)':>10}{'GEMM (ms)':>11}{'Exposed (ms)':>14}{'Efficiency':>12}")
    print(f"  {'─'*14} {'─'*9} {'─'*10} {'─'*12} {'─'*11}")
    for name, h2d, gemm in rows:
        gap = h2d - gemm
        eff = gemm / h2d * 100
        print(f"  {name:<15}{h2d:>10.3f}{gemm:>11.3f}{gap:>14.3f}{eff:>11.2f}%")
    print(f"{'━'*70}")
