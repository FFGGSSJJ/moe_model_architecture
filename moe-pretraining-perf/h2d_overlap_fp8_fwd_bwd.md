# FP8 H2D Overlap Benchmark: Forward + Backward Analysis

- **Hardware**: NVIDIA GH200 120GB
- **Date**: 2026-05-29
- **NUMA binding**: `numactl --cpunodebind=0 --membind=0`
- **Config parameters**:
    - C=CHUNK_SIZE (experts per transfer)
    - H=HIDDEN_SIZE
    - FFN=MOE_FFN_HIDDEN_SIZE
    - T=TOKEN_PER_EXPERT
- **Chunk size mapping**: FFN=1536 → C=4, FFN=2048 → C=4, FFN=4096 → C=2
- **FP8 peak**: 1979 TFLOPS (2× BF16 peak on GH200)
- **Benchmark script**: `bench_h2d_overlap_fp8.py` (swept via `run_fp8_sweep.py`)

---

## Table 1: H2D Transfer (Host-to-Device Async)

### H=4096

| Config | FC1 Latency (ms) | FC2 Latency (ms) | FC1 BW (GB/s) | FC2 BW (GB/s) | FC1 MBU (%) | FC2 MBU (%) |
|---|---|---|---|---|---|---|
| C=4, H=4096, FFN=1536, T=1024 | 0.124 | 0.065 | 404.57 | 389.72 | 89.90 | 86.60 |
| C=4, H=4096, FFN=2048, T=1024 | 0.162 | 0.083 | 413.33 | 405.81 | 91.85 | 90.18 |
| C=2, H=4096, FFN=4096, T=1024 | 0.162 | 0.083 | 415.29 | 405.30 | 92.29 | 90.07 |

### H=6144

| Config | FC1 Latency (ms) | FC2 Latency (ms) | FC1 BW (GB/s) | FC2 BW (GB/s) | FC1 MBU (%) | FC2 MBU (%) |
|---|---|---|---|---|---|---|
| C=4, H=6144, FFN=1536, T=1024 | 0.183 | 0.095 | 411.90 | 399.28 | 91.53 | 88.73 |
| C=4, H=6144, FFN=2048, T=1024 | 0.242 | 0.124 | 415.58 | 406.32 | 92.35 | 90.29 |
| C=2, H=6144, FFN=4096, T=1024 | 0.243 | 0.123 | 413.77 | 409.23 | 91.95 | 90.94 |

### H=7168

| Config | FC1 Latency (ms) | FC2 Latency (ms) | FC1 BW (GB/s) | FC2 BW (GB/s) | FC1 MBU (%) | FC2 MBU (%) |
|---|---|---|---|---|---|---|
| C=4, H=7168, FFN=1536, T=1024 | 0.214 | 0.109 | 411.97 | 404.19 | 91.55 | 89.82 |
| C=4, H=7168, FFN=2048, T=1024 | 0.281 | 0.143 | 417.44 | 409.38 | 92.76 | 90.97 |
| C=2, H=7168, FFN=4096, T=1024 | 0.282 | 0.143 | 416.15 | 411.14 | 92.48 | 91.37 |

NUMA binding achieves **~390–417 GB/s bandwidth** (~87–93% MBU of the 450 GB/s theoretical peak). H2D latency scales linearly with weight tensor size. Smaller FFN (1536) shows slightly lower MBU (~87–90%) due to smaller transfer sizes.

---

## Table 2: Forward GroupedGEMM Compute

### H=4096

| Config | FC1 Latency (ms) | FC2 Latency (ms) | FC1 TFLOPS | FC2 TFLOPS | FC1 MFU (%) | FC2 MFU (%) |
|---|---|---|---|---|---|---|
| C=4, H=4096, FFN=1536, T=1024 | 0.087 | 0.057 | 1182.07 | 903.24 | 59.73 | 45.64 |
| C=4, H=4096, FFN=2048, T=1024 | 0.124 | 0.069 | 1106.10 | 994.36 | 55.89 | 50.25 |
| C=2, H=4096, FFN=4096, T=1024 | 0.125 | 0.064 | 1100.86 | 1069.77 | 55.63 | 54.06 |

### H=6144

| Config | FC1 Latency (ms) | FC2 Latency (ms) | FC1 TFLOPS | FC2 TFLOPS | FC1 MFU (%) | FC2 MFU (%) |
|---|---|---|---|---|---|---|
| C=4, H=6144, FFN=1536, T=1024 | 0.128 | 0.074 | 1212.05 | 1039.10 | 61.25 | 52.51 |
| C=4, H=6144, FFN=2048, T=1024 | 0.179 | 0.093 | 1151.59 | 1113.46 | 58.19 | 56.26 |
| C=2, H=6144, FFN=4096, T=1024 | 0.182 | 0.086 | 1135.13 | 1193.05 | 57.36 | 60.29 |

### H=7168

| Config | FC1 Latency (ms) | FC2 Latency (ms) | FC1 TFLOPS | FC2 TFLOPS | FC1 MFU (%) | FC2 MFU (%) |
|---|---|---|---|---|---|---|
| C=4, H=7168, FFN=1536, T=1024 | 0.146 | 0.090 | 1231.48 | 1003.48 | 62.23 | 50.71 |
| C=4, H=7168, FFN=2048, T=1024 | 0.207 | 0.111 | 1163.32 | 1078.60 | 58.78 | 54.50 |
| C=2, H=7168, FFN=4096, T=1024 | 0.221 | 0.112 | 1087.15 | 1072.00 | 54.93 | 54.17 |

MFU = TFLOPS / 1979 (FP8 peak). FP8 GEMM achieves **46–62% MFU** across all configs.

---

## Table 3: Backward GroupedGEMM Compute

### H=4096

| Config | grad_a Latency (ms) | grad_x Latency (ms) | grad_a TFLOPS | grad_x TFLOPS | grad_a MFU (%) | grad_x MFU (%) |
|---|---|---|---|---|---|---|
| C=4, H=4096, FFN=1536, T=1024 | 0.047 | 0.096 | 1101.35 | 1074.38 | 55.65 | 54.29 |
| C=4, H=4096, FFN=2048, T=1024 | 0.064 | 0.124 | 1067.80 | 1109.63 | 53.96 | 56.07 |
| C=2, H=4096, FFN=4096, T=1024 | 0.064 | 0.121 | 1072.71 | 1139.96 | 54.20 | 57.60 |

### H=6144

| Config | grad_a Latency (ms) | grad_x Latency (ms) | grad_a TFLOPS | grad_x TFLOPS | grad_a MFU (%) | grad_x MFU (%) |
|---|---|---|---|---|---|---|
| C=4, H=6144, FFN=1536, T=1024 | 0.067 | 0.131 | 1162.00 | 1181.69 | 58.72 | 59.71 |
| C=4, H=6144, FFN=2048, T=1024 | 0.092 | 0.169 | 1118.48 | 1218.34 | 56.52 | 61.56 |
| C=2, H=6144, FFN=4096, T=1024 | 0.091 | 0.165 | 1128.13 | 1249.20 | 57.01 | 63.12 |

### H=7168

| Config | grad_a Latency (ms) | grad_x Latency (ms) | grad_a TFLOPS | grad_x TFLOPS | grad_a MFU (%) | grad_x MFU (%) |
|---|---|---|---|---|---|---|
| C=4, H=7168, FFN=1536, T=1024 | 0.076 | 0.158 | 1183.86 | 1144.27 | 59.82 | 57.82 |
| C=4, H=7168, FFN=2048, T=1024 | 0.107 | 0.205 | 1126.13 | 1175.71 | 56.90 | 59.41 |
| C=2, H=7168, FFN=4096, T=1024 | 0.114 | 0.202 | 1051.93 | 1190.90 | 53.15 | 60.18 |

- **grad_a**: `grad_s [M, ffn] = grad_y [M, hidden] @ w2.T [G, ffn, hidden]` — uses w2.T, same FLOPS as FWD FC2.
- **grad_x**: `dx [M, hidden] = grad_a [M, ffn*2] @ w1.T [G, hidden, ffn*2]` — uses w1.T, same FLOPS as FWD FC1.

BWD GEMM achieves **54–63% MFU**, comparable to FWD. grad_x (larger GEMM) tends toward higher MFU than grad_a.

---

## Table 4: Overlap Analysis (GEMM overlapped with H2D)

Overlap efficiency = GEMM_latency / H2D_latency (fraction of H2D hidden behind compute).
Exposed H2D = H2D_latency − GEMM_latency (the portion that cannot be overlapped).
BWD grad_a uses w2.T (same H2D size as FWD FC2). BWD grad_x uses w1.T (same H2D size as FWD FC1).

### H=4096

| Config | Layer | H2D (ms) | GEMM (ms) | Exposed (ms) | Efficiency (%) |
|---|---|---|---|---|---|
| C=4, FFN=1536 | FC1 (fwd) | 0.124 | 0.087 | 0.037 | 70.09 |
| | FC2 (fwd) | 0.065 | 0.057 | 0.008 | 88.36 |
| | grad_a (bwd) | 0.065 | 0.047 | 0.018 | 72.47 |
| | grad_x (bwd) | 0.124 | 0.096 | 0.028 | 77.12 |
| C=4, FFN=2048 | FC1 (fwd) | 0.162 | 0.124 | 0.038 | 76.53 |
| | FC2 (fwd) | 0.083 | 0.069 | 0.014 | 83.58 |
| | grad_a (bwd) | 0.083 | 0.064 | 0.019 | 77.83 |
| | grad_x (bwd) | 0.162 | 0.124 | 0.038 | 76.29 |
| C=2, FFN=4096 | FC1 (fwd) | 0.162 | 0.125 | 0.037 | 77.26 |
| | FC2 (fwd) | 0.083 | 0.064 | 0.019 | 77.59 |
| | grad_a (bwd) | 0.083 | 0.064 | 0.019 | 77.38 |
| | grad_x (bwd) | 0.162 | 0.121 | 0.041 | 74.61 |

### H=6144

| Config | Layer | H2D (ms) | GEMM (ms) | Exposed (ms) | Efficiency (%) |
|---|---|---|---|---|---|
| C=4, FFN=1536 | FC1 (fwd) | 0.183 | 0.128 | 0.055 | 69.60 |
| | FC2 (fwd) | 0.095 | 0.074 | 0.021 | 78.70 |
| | grad_a (bwd) | 0.095 | 0.067 | 0.028 | 70.37 |
| | grad_x (bwd) | 0.183 | 0.131 | 0.052 | 71.39 |
| C=4, FFN=2048 | FC1 (fwd) | 0.242 | 0.179 | 0.063 | 73.91 |
| | FC2 (fwd) | 0.124 | 0.093 | 0.031 | 74.74 |
| | grad_a (bwd) | 0.124 | 0.092 | 0.032 | 74.40 |
| | grad_x (bwd) | 0.242 | 0.169 | 0.073 | 69.86 |
| C=2, FFN=4096 | FC1 (fwd) | 0.243 | 0.182 | 0.061 | 74.65 |
| | FC2 (fwd) | 0.123 | 0.086 | 0.037 | 70.25 |
| | grad_a (bwd) | 0.123 | 0.091 | 0.032 | 74.29 |
| | grad_x (bwd) | 0.243 | 0.165 | 0.078 | 67.84 |

### H=7168

| Config | Layer | H2D (ms) | GEMM (ms) | Exposed (ms) | Efficiency (%) |
|---|---|---|---|---|---|
| C=4, FFN=1536 | FC1 (fwd) | 0.214 | 0.146 | 0.068 | 68.51 |
| | FC2 (fwd) | 0.109 | 0.090 | 0.019 | 82.49 |
| | grad_a (bwd) | 0.109 | 0.076 | 0.033 | 69.92 |
| | grad_x (bwd) | 0.214 | 0.158 | 0.056 | 73.73 |
| C=4, FFN=2048 | FC1 (fwd) | 0.281 | 0.207 | 0.074 | 73.49 |
| | FC2 (fwd) | 0.143 | 0.111 | 0.032 | 77.73 |
| | grad_a (bwd) | 0.143 | 0.107 | 0.036 | 74.45 |
| | grad_x (bwd) | 0.281 | 0.205 | 0.076 | 72.72 |
| C=2, FFN=4096 | FC1 (fwd) | 0.282 | 0.221 | 0.061 | 78.40 |
| | FC2 (fwd) | 0.143 | 0.112 | 0.031 | 78.55 |
| | grad_a (bwd) | 0.143 | 0.114 | 0.029 | 80.05 |
| | grad_x (bwd) | 0.282 | 0.202 | 0.080 | 71.57 |

---

## Key Takeaways

1. **Overlap efficiency is consistent between FWD and BWD** (68–88%). BWD grad_a has comparable efficiency to FWD FC2, and BWD grad_x has comparable efficiency to FWD FC1, since they share the same weight tensors (transposed) and identical FLOP counts.

2. **Exposed H2D per chunk**: 0.008–0.080 ms depending on config. For H=4096, exposed latency is **0.008–0.041 ms**; for H=6144 it is **0.021–0.078 ms**; for H=7168 it grows to **0.019–0.080 ms** due to larger weight tensors.

3. **FFN=1536 shows the lowest overlap efficiency for FC1/grad_x** (68–77%) but the highest for FC2/grad_a (79–88%). The smaller GEMM problem size reduces compute time faster than H2D time, making overlap harder for the larger FC1/grad_x GEMMs while making it easier for the smaller FC2/grad_a GEMMs.

4. **Overlap efficiency decreases with larger H** (H=4096: 70–88%, H=6144: 68–79%, H=7168: 68–82%) because H2D scales linearly with tensor size while GEMM throughput saturates.

5. **FC2/grad_a (smaller GEMM, smaller H2D) achieves the highest overlap efficiency** (70–88%), while FC1/grad_x (larger GEMM, larger H2D) shows lower efficiency (68–77%). The smaller absolute H2D of FC2 weights means a larger fraction is hidden behind compute.

6. **Total exposed H2D per MoE layer** (all 4 GEMM stages: FWD FC1 → FWD FC2 → BWD grad_a → BWD grad_x) sums to **~0.09–0.22 ms** depending on config. With 4 layers per pipeline stage, the total exposed H2D is **~0.36–0.88 ms**, which represents the irreducible overhead even with perfect chunk-level overlap.

---

## Per-Hidden-Size Overall Overlap Comparison

The table below shows the **total H2D and GEMM time per MoE layer** — i.e. the sum of all 4 GEMM stages (FWD FC1 → FWD FC2 → BWD grad_a → BWD grad_x), each of which requires its own weight H2D transfer for the next chunk. Overall efficiency = GEMM_total / H2D_total.

| Config | H2D Total (ms) | GEMM Total (ms) | Exposed (ms) | Overall Eff (%) |
|---|---|---|---|---|
| **H=4096** | | | | |
| FFN=1536, C=4 | 0.378 | 0.287 | 0.091 | 75.93 |
| FFN=2048, C=4 | 0.490 | 0.381 | 0.109 | 77.76 |
| FFN=4096, C=2 | 0.490 | 0.374 | 0.116 | 76.33 |
| **H=6144** | | | | |
| FFN=1536, C=4 | 0.556 | 0.400 | 0.156 | 71.94 |
| FFN=2048, C=4 | 0.732 | 0.533 | 0.199 | 72.81 |
| FFN=4096, C=2 | 0.732 | 0.524 | 0.208 | 71.58 |
| **H=7168** | | | | |
| FFN=1536, C=4 | 0.646 | 0.470 | 0.176 | 72.76 |
| FFN=2048, C=4 | 0.848 | 0.630 | 0.218 | 74.29 |
| FFN=4096, C=2 | 0.850 | 0.649 | 0.201 | 76.35 |

### Summary by Hidden Size

| Hidden Size | Avg Efficiency (%) | Eff Range (%) | Avg Exposed (ms) | Avg Exposed per Layer×4 (ms) |
|---|---|---|---|---|
| H=4096 | 76.7 | 75.9–77.8 | 0.105 | 0.421 |
| H=6144 | 72.1 | 71.6–72.8 | 0.188 | 0.751 |
| H=7168 | 74.5 | 72.8–76.4 | 0.198 | 0.793 |

### Analysis

1. **H=4096 achieves the highest overall overlap efficiency (~77%)**. The smaller weight tensors keep H2D time low relative to GEMM time, hiding ~77% of transfers behind compute. Exposed H2D per layer is only ~0.1 ms, or ~0.4 ms for a 4-layer pipeline stage.

2. **H=6144 has the lowest overall efficiency (~72%)**. It sits in a "valley" where H2D has grown significantly (weight tensors are 1.5× larger than H=4096) but the GEMM problem sizes have not yet scaled enough to fully compensate. Exposed H2D nearly doubles to ~0.19 ms per layer.

3. **H=7168 partially recovers to ~74.5% efficiency**. Despite having the largest weight tensors, the GEMM problem sizes are now large enough to achieve higher MFU (~55–63%), which improves compute-to-transfer ratio. The FFN=4096 variant at H=7168 reaches 76.4% — close to H=4096 levels — because its large FC2 GEMM (H×FFN = 7168×4096) keeps the GPU well-utilized.

4. **Exposed H2D scales roughly linearly with H**: 0.1 ms (H=4096) → 0.19 ms (H=6144) → 0.20 ms (H=7168). The jump from 4096 to 6144 (+80%) is much larger than from 6144 to 7168 (+5%), confirming that H=6144's lower efficiency is primarily driven by the H2D cost increase not being offset by proportionally faster GEMM.

5. **Practical impact**: For a 4-layer MoE pipeline stage, the total irreducible exposed H2D latency ranges from **~0.42 ms (H=4096)** to **~0.79 ms (H=7168)**. At typical iteration times of ~1–2 s, this represents ~0.02–0.08% overhead — negligible in practice for all configs.
