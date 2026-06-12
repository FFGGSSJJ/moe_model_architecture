# EP Overlap Microbenchmark

## Motivation

`--overlap-moe-expert-parallel-comm` hides the MoE All-to-All (A2A) behind compute. We want
to quantify, **in isolation**, how much of the EP A2A can be covered by expert-MLP
(GroupedGEMM) compute on our GH200 cluster, and where the break-even point sits. This doc is
the plan; the first deliverable below bounds the A2A *communication volume per device*, which
is the quantity the overlap has to hide.

## Preliminary

We reuse the symbols from [`why-offloading-experts.md`](why-offloading-experts.md) and
[`offload-experts.md`](offload-experts.md).

**Model**

| Symbol      | Def                                                            |
| ----------- | ------------------------------------------------------------- |
| $N_e$       | Total number of routed experts                                |
| $N_a$       | Activated routed experts per token (top-$K$)                  |
| $H$         | Hidden size — the size of a *dispatched* token vector         |
| $h_e$       | Intermediate size of an expert                                |
| $L_{moe}$   | Number of MoE layers                                          |
| $M$         | Tokens assigned to each expert in the balanced scenario       |
| $b$         | Bytes per element of the dispatch payload (BF16 = 2, FP8 = 1) |

**Parallelism / system**

| Symbol           | Def                                                        |
| ---------------- | ---------------------------------------------------------- |
| $N_{gpu}$        | World size                                                 |
| $TP, PP, EP$     | Tensor / pipeline / expert parallel size                  |
| $ETP$            | Expert tensor parallel size (**$=1$** in our setup)        |
| $G$              | GPUs per node ($=4$ on the Alps GH200 nodes)              |
| $DP$             | $N_{gpu}/(TP\cdot PP)$                                     |
| $EDP$            | $N_{gpu}/(EP\cdot ETP\cdot PP) = N_{gpu}/(EP\cdot PP)$     |
| $\beta_\text{inter}$ | Inter-node A2A bandwidth, $\approx 25$ GB/s (measured) |
| $\beta_\text{intra}$ | Intra-node (NVLink) A2A bandwidth, $\gg \beta_\text{inter}$ |

Balanced tokens-per-expert (carried over from the repo):

$$M = \frac{mbs\cdot seq\cdot N_a}{N_e}\cdot\frac{DP}{EDP} = mbs\cdot seq\cdot\frac{EP}{TP}\cdot\frac{N_a}{N_e}$$

**Assumptions for this analysis**

1. **Balanced routing** (each expert receives exactly $M$ tokens). This is the regime the
   `--force-load-balancing` perf tests target, and the *best case* for the dispatcher.
2. **Sequence parallel on**: each rank feeds $mbs\cdot seq/TP$ tokens into the MoE layer.
3. **$ETP = 1$**: experts are sharded by EP only (consistent with $DP/EDP = EP/TP$).
4. **EP ranks are node-contiguous**: the first $g \equiv \min(EP, G)$ EP ranks share a node, so
   $g-1$ peers are reachable over NVLink and $EP-g$ peers only over Slingshot.

## EP All-to-All Communication Volume (per device)

### 1. Token flow

One MoE layer issues two A2A collectives in the forward pass:

```
router → [dispatch A2A] → expert GroupedGEMM → [combine A2A] → (weighted sum)
```

The backward pass mirrors this: the gradient of `combine` is a dispatch-shaped A2A and the
gradient of `dispatch` is a combine-shaped A2A. So per MoE layer there are **2 A2As in forward
and 4 across fwd+bwd**, all of the same volume.

### 2. Per-device payload

After dispatch, a rank holds the tokens routed to the $N_e/EP$ experts it hosts. In the
balanced case that is

$$\tau \;\equiv\; \underbrace{\frac{N_e}{EP}}_{\text{experts/rank}}\cdot\, M \;=\; \frac{mbs\cdot seq\cdot N_a}{TP}\quad\text{tokens.}$$

Equivalently, the rank's own input is $mbs\cdot seq/TP$ tokens, each replicated $N_a$ times by
top-$K$ routing, giving the same $\tau$ token-copies it must *send*. (Conservation: in the
balanced case send = receive = $\tau$.) Each copy is an $H$-vector of $b$ bytes, so the rank's
full dispatch buffer is

$$\boxed{\,D \;=\; \tau\,H\,b \;=\; \frac{mbs\cdot seq\cdot N_a}{TP}\,H\,b \quad\text{bytes.}\,}$$

Of this, the $1/EP$ fraction destined for *local* experts never leaves the rank. The volume
that actually crosses the A2A (sent, and by symmetry received) is

$$V_\text{disp} \;=\; D\cdot\frac{EP-1}{EP}.$$

Combine moves the processed tokens back the same way, so $V_\text{comb} = V_\text{disp}$. Per
MoE layer:

$$V_\text{fwd} = 2D\,\frac{EP-1}{EP}, \qquad V_\text{step (fwd+bwd)} = 4D\,\frac{EP-1}{EP}.$$

### 3. Intra- vs. inter-node split (where 25 GB/s bites)

Splitting the wire volume by link, with $g = \min(EP, G)$:

$$V_\text{intra} = D\,\frac{g-1}{EP}\ \text{(NVLink)}, \qquad V_\text{inter} = D\,\frac{EP-g}{EP}\ \text{(Slingshot)}.$$

Because $\beta_\text{inter} \ll \beta_\text{intra}$, the A2A latency is set by the inter-node
leg. The exposed dispatch time per layer is

$$T_\text{disp} \;\approx\; \frac{V_\text{inter}}{\beta_\text{inter}} \;=\; \frac{D}{\beta_\text{inter}}\cdot\frac{EP-g}{EP},
\qquad T_\text{fwd} = 2\,T_\text{disp}.$$

Two regimes fall straight out:

- **$EP \le G$** (e.g. EP4): $V_\text{inter}=0$, the whole A2A is on NVLink — cheap, easy to
  overlap. This is why EP4 is the comfortable point on this cluster.
- **$EP > G$** (EP8/16/32): the inter-node fraction $\frac{EP-g}{EP}$ grows toward 1 and
  $T_\text{disp}$ is dominated by the 25 GB/s link. Larger EP shrinks $D$ (more ranks share the
  load) but raises the inter-node *fraction*, and the latter wins past EP8 — matching the
  observed "EP16 is completely communication dominated."

## Microbenchmark Design

### Objectives

1. Measure **dispatch** (and **combine**) A2A latency as a function of EP $\in\{8,16,32\}$ and
   the model dimensions $H$, $h_e$, $N_e$, $N_a$, precision.
2. Extract the **effective inter-node bandwidth** $\beta_\text{eff}$ and the **fixed overhead**
   $\alpha$ (kernel launch + latency floor), and check the Step-1 prediction
   $T \approx \alpha + V_\text{inter}/\beta_\text{eff}$.
3. Produce the $T_\text{A2A}$ numbers that the overlap analysis (next doc) divides into
   $T_\text{gemm}$ to get overlap efficiency $\eta = T_\text{gemm}/T_\text{A2A}$.

### What actually moves dispatch latency (read before sweeping)

From Step 1, the per-device dispatch volume is

$$D = \tau\,H\,b,\qquad \tau=\frac{mbs\cdot seq\cdot N_a}{TP},\qquad T_\text{disp}\approx\alpha+\frac{D}{\beta_\text{eff}}\cdot\frac{EP-g}{EP}.$$

So the raw A2A latency depends **only** on $(\tau, H, b)$ and the inter-node fraction
$\frac{EP-g}{EP}$ — i.e. on $N_a$, $H$, precision, and $EP$. It is **independent of $N_e$ and
$h_e$.** That makes the requested sweep purposeful rather than redundant:

| Knob you sweep | Effect on **raw** A2A (Tier A) | Why sweep it anyway |
| -------------- | ------------------------------ | ------------------- |
| $H$            | message size $\propto H$ — **linear** | primary scaling axis; fit $\beta_\text{eff}$ |
| $N_a$ (or token count) | $\tau\propto N_a$ — **linear** | second volume axis; cross-check $\beta_\text{eff}$ |
| $EP\in\{8,16,32\}$ | inter-node fraction $\frac{EP-g}{EP}$, peer count | the headline axis (multi-node) |
| precision      | $b$: BF16 $\to$ FP8 halves bytes | quantifies the FP8-dispatch win |
| $N_e$          | **none on bytes** (flat) | isolates **permutation/grouping overhead** (Tier B); confirms A2A is $N_e$-invariant |
| $h_e$          | **none** on dispatch | only feeds the overlap GEMM (Step 2); keep at production values |

> Practical consequence: sweep $H$, $N_a$, $EP$, precision to characterize the **communication**;
> sweep $N_e$ (and later $h_e$) to characterize **software overhead** and the **compute** side.

### Two-tier harness

- **Tier A — raw NCCL A2A (ceiling).** A standalone `torch.distributed.all_to_all_single` over
  a world of size $EP$. Pure NCCL; no permute, no quantize. Gives the hardware ceiling and the
  $\beta_\text{eff}$ used in Step 1.
- **Tier B — real dispatcher (exposed cost).** Drive Megatron's `MoEAlltoAllTokenDispatcher`
  (`token_permutation` for dispatch, `token_unpermutation` for combine) with a mock balanced
  routing map. Captures permute-fusion, padding, and FP8 quantization. The gap **Tier B − Tier A**
  is the software overhead, and it is the part that grows with $N_e$.

### Topology & launch (replicate production)

To make $g=\min(EP,G)$ match production, lay EP ranks out **node-contiguous at $G=4$ GPUs/node**:

| EP  | Nodes | GPUs/node | $g$ (intra) | inter-node frac |
| :-: | :---: | :-------: | :---------: | :-------------: |
|  8  |   2   |     4     |      4      |      0.50       |
| 16  |   4   |     4     |      4      |      0.75       |
| 32  |   8   |     4     |      4      |      0.875      |

- World size $= EP$; one EP rank per GPU; rank $i$ on node $\lfloor i/4\rfloor$.
- `numactl --cpunodebind=0 --membind=0` (as in the H2D bench).
- Same NCCL env that achieved 25 GB/s (the "NCCL env var fixes" from
  [`why-offloading-experts.md`](why-offloading-experts.md)); pin
  `NCCL_ALGO`, `NCCL_IB/Slingshot` vars and record them with the results.
- Launch with `srun`/`torchrun`; one `--ntasks-per-node 4`.

### Sweeps (one-factor-at-a-time around an anchor)

**Anchor** = the 670B-A40B target: $H{=}7168$, $h_e{=}4096$, $N_e{=}128$, $N_a{=}4$, $mbs{=}2$,
$seq{=}4096$, $TP{=}4$, BF16, $EP{=}16$ ⟹ $\tau=8192$. All predictions below use
$\beta_\text{eff}=25$ GB/s, $\alpha=0$ (decimal GB), per **MoE layer dispatch only**
($T_\text{fwd}=2T_\text{disp}$). Require $\tau \bmod EP = 0$ for balanced splits.

**S1 — Hidden size** ($EP{=}16$, $N_a{=}4$ ⟹ $\tau{=}8192$, BF16)

| $H$  | $D$ (GB) | $V_\text{inter}$ (GB) | predicted $T_\text{disp}$ (ms) |
| :--: | :------: | :-------------------: | :----------------------------: |
| 2048 |  0.0336  |        0.0252         |             1.01               |
| 4096 |  0.0671  |        0.0503         |             2.01               |
| 7168 |  0.1174  |        0.0881         |             3.52               |
| 8192 |  0.1342  |        0.1007         |             4.03               |

**S2 — Activated experts / token count** ($EP{=}16$, $H{=}7168$, BF16)

| $N_a$ | $\tau$ | $D$ (GB) | $V_\text{inter}$ (GB) | predicted $T_\text{disp}$ (ms) |
| :---: | :----: | :------: | :-------------------: | :----------------------------: |
|   4   |  8192  |  0.1174  |        0.0881         |             3.52               |
|   8   | 16384  |  0.2348  |        0.1761         |             7.04               |
|  14   | 28672  |  0.4110  |        0.3082         |            12.33               |
|  28   | 57344  |  0.8220  |        0.6165         |            24.66               |

**S3 — Expert count** ($EP{=}16$, $H{=}7168$, $N_a{=}4$ ⟹ $\tau{=}8192$, BF16) — **A2A bytes
invariant**; only $M=\tau\,EP/N_e$ (GEMM granularity) and Tier-B permute cost change.

| $N_e$ | $M$ (tokens/expert) | $D$ (GB) | predicted Tier-A $T_\text{disp}$ (ms) | Tier-B expectation |
| :---: | :-----------------: | :------: | :-----------------------------------: | ------------------ |
|  64   |        2048         |  0.1174  |                3.52                    | flat A2A; lowest permute |
| 128   |        1024         |  0.1174  |                3.52                    | flat A2A           |
| 256   |         512         |  0.1174  |                3.52                    | permute ↑          |
| 448   |         293         |  0.1174  |                3.52                    | permute ↑↑ (more bins) |

**S4 — EP size** ($H{=}7168$, $N_a{=}4$ ⟹ $\tau{=}8192$, BF16) — $D$ invariant; inter-node
fraction is the only mover. **The headline sweep.**

| EP  | inter frac | $V_\text{inter}$ (GB) | predicted $T_\text{disp}$ (ms) |
| :-: | :--------: | :-------------------: | :----------------------------: |
|  8  |   0.50     |        0.0587         |             2.35               |
| 16  |   0.75     |        0.0881         |             3.52               |
| 32  |   0.875    |        0.1028         |             4.11               |

**S5 — Precision** ($EP{=}16$, $H{=}7168$, $N_a{=}4$): BF16 $b{=}2 \Rightarrow 3.52$ ms vs
FP8 $b{=}1 \Rightarrow 1.76$ ms. (FP8 A2A needs NCCL $\ge$ 2.28; if unavailable, proxy the wire
size with an `int8`/byte tensor to measure the volume effect, and benchmark the TE quantize cost
separately in Tier B.)

### Timing methodology

- **Warm-up** $\ge$ 20 iters (first NCCL call builds the ring/channels and is an outlier).
- Time each iter with **CUDA events**; `dist.barrier()` immediately before `start.record()` so
  ranks enter together.
- $\ge$ 50 timed iters. The collective finishes when the **slowest** rank finishes, so reduce as
  **per-iter `max` across ranks**, then report **median** and **p90** over iters (+ std).
- Report alongside latency:
  - **Effective inter-node BW** $= V_\text{inter}/T$ (GB/s) → compare to the 25 GB/s ceiling.
  - **NCCL bus bandwidth** for all-to-all: $\text{algbw}=\text{send\_bytes}/T$,
    $\text{busbw}=\text{algbw}\cdot\frac{EP-1}{EP}$ (for cross-check with `nccl-tests`).
  - For Tier B: **permute overhead** $=T_\text{TierB}-T_\text{TierA}$.

### Hypotheses to confirm/refute (so the data has a verdict)

- **H1**: $T_\text{disp}$ is **linear in $H$ and in $\tau$** with a single $\beta_\text{eff}$
  (S1, S2 collapse onto one line vs $V_\text{inter}$); intercept $\alpha$ = launch/latency floor.
- **H2**: across S4, $\beta_\text{eff}$ stays $\approx$ constant while latency rises only through
  $\frac{EP-g}{EP}$ — confirming EP8→16→32 degradation is the **inter-node fraction**, not raw
  volume (the basis for "constrain to EP4/EP8").
- **H3**: Tier-A latency is **flat in $N_e$** (S3); any rise is Tier-B permute overhead.
- **H4**: FP8 dispatch ≈ **halves** $T_\text{disp}$ (S5), bounding the upside seen in the +13%
  end-to-end FP8-dispatch result.

### Pitfalls

- **NIC contention**: with 4 GPUs/node all injecting inter-node, confirm the per-GPU 25 GB/s
  holds (GH200 = 1 NIC/GPU). Optionally sweep GPUs/node $\in\{1,2,4\}$ to isolate contention.
- **Imbalance**: all predictions assume `--force-load-balancing`. A realistic imbalanced routing
  map (hot experts) lengthens the slowest rank and inflates $T$; run it as a secondary sweep to
  bound the worst case.
- **In-place reuse**: don't reuse the same send/recv buffers in a way that lets the allocator or
  cache hide real traffic; allocate fresh or rotate buffers.
- **Clock alignment**: without the pre-record barrier, CUDA-event latency includes skew and looks
  artificially long.

### Pseudo-code skeleton (Tier A)

```python
# world_size == EP; launched 4 ranks/node, node-contiguous
import torch, torch.distributed as dist
dist.init_process_group("nccl")
EP, rank = dist.get_world_size(), dist.get_rank()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def bench_dispatch(tau, H, dtype, iters=50, warmup=20):
    assert tau % EP == 0
    send = torch.randn(tau, H, device="cuda", dtype=dtype)
    recv = torch.empty_like(send)
    splits = [tau // EP] * EP                      # balanced
    for _ in range(warmup):
        dist.all_to_all_single(recv, send, splits, splits)
    torch.cuda.synchronize(); dist.barrier()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True); ts = []
    for _ in range(iters):
        dist.barrier(); s.record()
        dist.all_to_all_single(recv, send, splits, splits)
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))               # ms, this rank
    t = torch.tensor(ts, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)       # slowest rank per iter
    return t.median().item(), t.quantile(0.9).item()
# sweep: for H in S1 / for tau in S2 / for EP in S4 / dtype in {bf16, fp8/int8}
```

## Roadmap

1. **This doc**: A2A communication volume + dispatch-latency microbench plan (Tier A/B, S1–S5).
2. **Compute side**: per-device expert-MLP $T_\text{gemm}$ (GroupedGEMM, gated $H\!\to\!2h_e\!\to\!H$)
   and overlap efficiency $\eta=T_\text{gemm}/T_\text{A2A}$ — mirror of the H2D metric in
   [`h2d_overlap_bf16_vs_fp8.md`](h2d_overlap_bf16_vs_fp8.md).
3. **Overlapped runs**: A2A issued on a side stream concurrently with GroupedGEMM, reported in the
   same 3-table layout (isolated A2A / isolated GEMM / overlapped) as the H2D doc.
