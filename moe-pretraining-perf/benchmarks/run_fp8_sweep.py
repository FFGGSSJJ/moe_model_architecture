"""
Sweep FP8 overlap benchmarks (H2D + FWD + BWD) across 4 model configs.

Uses bench_h2d_overlap_fp8.py by patching its module-level constants.
Outputs results as JSON for downstream processing.
"""

import sys
import os
import json

# ---------------------------------------------------------------------------
# Configs to sweep (chunk_size, hidden, ffn, tokens_per_expert)
#   Chunk size mapping: FFN=2048 -> C=4, FFN=4096 -> C=2
# ---------------------------------------------------------------------------
CONFIGS = [
    (4, 4096, 1536, 1024),
    (4, 4096, 2048, 1024),
    (2, 4096, 4096, 1024),
    (4, 7168, 1536, 1024),
    (4, 7168, 2048, 1024),
    (2, 7168, 4096, 1024),
]


def run_config(chunk_size, hidden, ffn, tok):
    """Patch bench_h2d_overlap_fp8 module globals and run all benchmarks."""
    import bench_h2d_overlap_fp8 as bench

    bench.CHUNK_SIZE = chunk_size
    bench.HIDDEN_SIZE = hidden
    bench.MOE_FFN_HIDDEN_SIZE = ffn
    bench.TOKEN_PER_EXPERT = tok

    tag = f"C={chunk_size}, H={hidden}, FFN={ffn}, T={tok}"
    print(f"\n{'━'*70}")
    print(f"  Config: {tag}")
    print(f"{'━'*70}")

    # H2D
    print(f"  H2D...", end=" ", flush=True)
    fc1_h2d, fc2_h2d = bench.bench_h2d()
    print(f"FC1={fc1_h2d:.3f}ms  FC2={fc2_h2d:.3f}ms")

    # FWD GEMM
    print(f"  FWD GEMM...", end=" ", flush=True)
    fc1_fwd, fc2_fwd = bench.bench_fp8_groupgemm()
    print(f"FC1={fc1_fwd:.3f}ms  FC2={fc2_fwd:.3f}ms")

    # BWD grad_a
    print(f"  BWD grad_a...", end=" ", flush=True)
    bwd_grad_a = bench.bench_fp8_groupgemm_bwd_grad_a()
    print(f"{bwd_grad_a:.3f}ms")

    # BWD grad_x
    print(f"  BWD grad_x...", end=" ", flush=True)
    bwd_grad_x = bench.bench_fp8_groupgemm_bwd_grad_x()
    print(f"{bwd_grad_x:.3f}ms")

    return {
        "config": {
            "chunk_size": chunk_size,
            "hidden": hidden,
            "ffn": ffn,
            "tokens_per_expert": tok,
        },
        "h2d": {
            "fc1_ms": fc1_h2d,
            "fc2_ms": fc2_h2d,
        },
        "fwd": {
            "fc1_ms": fc1_fwd,
            "fc2_ms": fc2_fwd,
        },
        "bwd": {
            "grad_a_ms": bwd_grad_a,
            "grad_x_ms": bwd_grad_x,
        },
    }


def compute_derived(r):
    """Compute TFLOPS, MFU, overlap metrics for a result."""
    c = r["config"]
    C, H, FFN, T = c["chunk_size"], c["hidden"], c["ffn"], c["tokens_per_expert"]
    total_tok = C * T

    # FWD FLOPS
    fc1_flops = C * 2.0 * T * H * FFN * 2
    fc2_flops = C * 2.0 * T * FFN * H
    # BWD FLOPS (same as corresponding FWD)
    grad_a_flops = fc2_flops  # grad_y[M,H] @ w2.T[G,FFN,H] same as FWD FC2
    grad_x_flops = fc1_flops  # grad_a[M,FFN*2] @ w1.T[G,H,FFN*2] same as FWD FC1

    r["fwd"]["fc1_tflops"] = fc1_flops / (r["fwd"]["fc1_ms"] * 1e-3) / 1e12
    r["fwd"]["fc2_tflops"] = fc2_flops / (r["fwd"]["fc2_ms"] * 1e-3) / 1e12
    r["bwd"]["grad_a_tflops"] = grad_a_flops / (r["bwd"]["grad_a_ms"] * 1e-3) / 1e12
    r["bwd"]["grad_x_tflops"] = grad_x_flops / (r["bwd"]["grad_x_ms"] * 1e-3) / 1e12

    # MFU (FP8 peak = 1979 TFLOPS on GH200)
    fp8_peak = 1979.0
    r["fwd"]["fc1_mfu"] = r["fwd"]["fc1_tflops"] / fp8_peak * 100
    r["fwd"]["fc2_mfu"] = r["fwd"]["fc2_tflops"] / fp8_peak * 100
    r["bwd"]["grad_a_mfu"] = r["bwd"]["grad_a_tflops"] / fp8_peak * 100
    r["bwd"]["grad_x_mfu"] = r["bwd"]["grad_x_tflops"] / fp8_peak * 100

    # Overlap: grad_a uses w2.T (same H2D as FC2), grad_x uses w1.T (same H2D as FC1)
    r["overlap"] = {
        "fwd_fc1_eff": r["fwd"]["fc1_ms"] / r["h2d"]["fc1_ms"] * 100,
        "fwd_fc2_eff": r["fwd"]["fc2_ms"] / r["h2d"]["fc2_ms"] * 100,
        "bwd_grad_a_eff": r["bwd"]["grad_a_ms"] / r["h2d"]["fc2_ms"] * 100,
        "bwd_grad_x_eff": r["bwd"]["grad_x_ms"] / r["h2d"]["fc1_ms"] * 100,
    }

    return r


if __name__ == "__main__":
    # Make sure imports work from the benchmarks directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    all_results = []
    for chunk_size, hidden, ffn, tok in CONFIGS:
        r = run_config(chunk_size, hidden, ffn, tok)
        r = compute_derived(r)
        all_results.append(r)

    # Save JSON
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fp8_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")
