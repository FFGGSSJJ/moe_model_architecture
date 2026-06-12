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

## Microbenchmark A — A2A Dispatch Latency

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

So the A2A latency depends **only** on $(\tau, H, b)$ and the inter-node fraction
$\frac{EP-g}{EP}$ — i.e. on $N_a$, $H$, precision, and $EP$. It is **independent of $N_e$ and
$h_e$.** That focuses the sweep:

| Knob you sweep | Effect on A2A latency | Why sweep it |
| -------------- | --------------------- | ------------ |
| $H$            | message size $\propto H$ — **linear** | primary scaling axis; fit $\beta_\text{eff}$ |
| $N_a$ (or token count) | $\tau\propto N_a$ — **linear** | second volume axis; cross-check $\beta_\text{eff}$ |
| $EP\in\{8,16,32\}$ | inter-node fraction $\frac{EP-g}{EP}$, peer count | the headline axis (multi-node) |
| precision      | $b$: BF16 $\to$ FP8 halves bytes | quantifies the FP8-dispatch win |
| $N_e$          | **none on bytes** (flat) | control: confirms A2A is $N_e$-invariant |
| $h_e$          | **none** on dispatch | belongs to the compute side (Microbenchmark B); hold at production value |

> Practical consequence: $H$, $N_a$, $EP$, and precision are the axes that move A2A latency;
> $N_e$ is a flat control and $h_e$ has no effect here.

### Harness

A standalone `torch.distributed.all_to_all_single` over a world of size $EP$ — pure NCCL, no
permute, no quantize. This isolates the collective itself and yields the effective inter-node
bandwidth $\beta_\text{eff}$ used in Step 1.

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
$seq{=}4096$, $TP{=}4$, BF16, $EP{=}16$ ⟹ $\tau=8192$. All predictions below use $\beta_\text{eff}=25$ GB/s, $\alpha=0$ (decimal GB), per **MoE layer dispatch only** ($T_\text{fwd}=2T_\text{disp}$). Require $\tau \bmod EP = 0$ for balanced splits.

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

**S3 — Expert count (control)** ($EP{=}16$, $H{=}7168$, $N_a{=}4$ ⟹ $\tau{=}8192$, BF16) — **A2A
bytes are invariant** in $N_e$; only $M=\tau\,EP/N_e$ (the per-expert GEMM size, used by
Microbenchmark B) changes. Expect a flat line — a sanity check on the harness.

| $N_e$ | $M$ (tokens/expert) | $D$ (GB) | predicted $T_\text{disp}$ (ms) |
| :---: | :-----------------: | :------: | :----------------------------: |
|  64   |        2048         |  0.1174  |              3.52              |
| 128   |        1024         |  0.1174  |              3.52              |
| 256   |         512         |  0.1174  |              3.52              |
| 448   |         293         |  0.1174  |              3.52              |

**S4 — EP size** ($H{=}7168$, $N_a{=}4$ ⟹ $\tau{=}8192$, BF16) — $D$ invariant; inter-node
fraction is the only mover. **The headline sweep.**

| EP  | inter frac | $V_\text{inter}$ (GB) | predicted $T_\text{disp}$ (ms) |
| :-: | :--------: | :-------------------: | :----------------------------: |
|  8  |   0.50     |        0.0587         |             2.35               |
| 16  |   0.75     |        0.0881         |             3.52               |
| 32  |   0.875    |        0.1028         |             4.11               |

**S5 — Precision** ($EP{=}16$, $H{=}7168$, $N_a{=}4$): BF16 $b{=}2 \Rightarrow 3.52$ ms vs
FP8 $b{=}1 \Rightarrow 1.76$ ms. (FP8 A2A needs NCCL $\ge$ 2.28; if unavailable, proxy the wire
size with an `int8`/byte tensor to measure the volume effect.)

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

### Hypotheses to confirm/refute (so the data has a verdict)

- **H1**: $T_\text{disp}$ is **linear in $H$ and in $\tau$** with a single $\beta_\text{eff}$
  (S1, S2 collapse onto one line vs $V_\text{inter}$); intercept $\alpha$ = launch/latency floor.
- **H2**: across S4, $\beta_\text{eff}$ stays $\approx$ constant while latency rises only through
  $\frac{EP-g}{EP}$ — confirming EP8→16→32 degradation is the **inter-node fraction**, not raw
  volume (the basis for "constrain to EP4/EP8").
- **H3**: A2A latency is **flat in $N_e$** (S3) — a control confirming the collective cost is set
  by bytes, not expert count.
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

### Pseudo-code skeleton

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

## Microbenchmark B — GroupedGEMM × H2D Overlap (per device)

### Why a second benchmark

[`h2d_overlap_bf16_vs_fp8.md`](h2d_overlap_bf16_vs_fp8.md) measures **one chunk** of $C$ experts
at a **fixed** $T=1024$ tokens/expert. It does not vary the two knobs that decide whether expert
offloading actually hides behind compute at scale:

1. **EP — through $M$.** The per-expert token count is $M = mbs\cdot seq\cdot\frac{EP}{TP}\cdot\frac{N_a}{N_e}$,
   so EP sets the GEMM's free dimension and hence the overlap ratio. $T=1024$ is just one EP point.
2. **Per-device pipeline depth.** A rank owns $n_{exp}=N_e/EP$ experts, streamed in
   $n_{chunk}=\lceil n_{exp}/C\rceil$ chunks. Pipeline fill/drain (the first load and last compute
   cannot be hidden) matters, and $n_{chunk}$ **shrinks with EP** — the opposite pull to (1).

This benchmark drives both $M$ and $n_{chunk}$ from the parallel config and measures the **whole
per-device load→compute pipeline**, not a single chunk.

### Per-device model (gated MLP, forward)

Per rank: $n_{exp}=N_e/EP$ experts $\times\ M$ tokens. Note $n_{exp}\cdot M=\tau=\frac{mbs\cdot seq\cdot N_a}{TP}$
— the **same** $\tau$ that drives the A2A in Microbenchmark A.

Per expert (gated $H\!\to\!2h_e\!\to\!H$):

- weights: $\underbrace{H\cdot 2h_e}_{\text{FC1}}+\underbrace{h_e\cdot H}_{\text{FC2}}=3Hh_e$ elements $\Rightarrow 3Hh_e\,b$ bytes
- FLOPs (fwd): $\underbrace{4MHh_e}_{\text{FC1}}+\underbrace{2MHh_e}_{\text{FC2}}=6MHh_e$

Per chunk of $C$ experts ($\beta_\text{H2D}=450$ GB/s):

$$T_\text{load}=\frac{C\cdot 3Hh_e\,b}{\beta_\text{H2D}\cdot \text{MBU}},\qquad T_\text{gemm}=\frac{C\cdot 6MHh_e}{\text{FLOPS}\cdot \text{MFU}}.$$

The per-chunk overlap efficiency at peak (MBU=MFU=1) collapses to the repo's formula —
**independent of $H,h_e,C$**:

$$\eta=\frac{T_\text{gemm}}{T_\text{load}}=\frac{2M}{b}\cdot\frac{\beta_\text{H2D}}{\text{FLOPS}}\ \xrightarrow{\text{BF16}}\ M\cdot\frac{450}{989\text{e}3}=M\cdot 4.55\text{e}{-4}.$$

(FP8 doubles FLOPS and halves $b$ ⟹ the **analytical** $\eta$ is unchanged; the **realized** one
differs through MFU/MBU.) Reaching $\eta=1$ needs $M\approx2200$ (ideal) or $\approx1465$ at
90% MBU / 60% MFU — consistent with [`offload-experts.md`](offload-experts.md); target $M\gtrsim1000$.

### EP is a *shape* knob, not a *volume* knob

Per rank, across EP (fixed model/TP):

| Quantity | Scaling in EP | Meaning |
| -------- | :-----------: | ------- |
| GEMM FLOPs $=6Hh_e\,\tau$ | **flat** | total compute / rank fixed |
| H2D bytes $=n_{exp}\cdot 3Hh_e\,b$ | $\propto 1/EP$ | fewer experts/rank ⟹ less weight to load |
| $M$ (GEMM free dim) | $\propto EP$ | fatter per-expert GEMM ⟹ higher MFU, higher $\eta$ |
| $n_{chunk}=\lceil (N_e/EP)/C\rceil$ | $\propto 1/EP$ | shallower pipeline ⟹ worse fill/drain |

So **larger EP makes offloading easier** (less H2D/rank, higher per-chunk $\eta$) — the *opposite*
of what it does to the A2A, where larger EP raises the inter-node fraction (Microbenchmark A).
The two benchmarks bracket the EP trade-off from each side.

### Per-device pipeline timing

Double-buffer: H2D of chunk $i{+}1$ on a copy stream while GroupedGEMM of chunk $i$ runs on the
compute stream. End-to-end MLP latency for $n=n_{chunk}$ balanced chunks:

$$T_\text{dev}\ \approx\ \underbrace{T_\text{load}}_{\text{prologue}}\ +\ (n-1)\max(T_\text{gemm},T_\text{load})\ +\ \underbrace{T_\text{gemm}}_{\text{epilogue}}.$$

Device-level metrics:

- **Exposed H2D** $= T_\text{dev}-n\,T_\text{gemm}$ — what offloading costs vs. weights-resident.
- **Device overlap efficiency** $= \dfrac{n\,T_\text{gemm}}{T_\text{dev}}$ — strictly **below** the
  per-chunk $\eta$ when $n$ is small (prologue weight). At EP32, $C{=}2$ ⟹ $n{=}2$: the unhideable
  prologue is half the loads.

### Sweeps (anchor = 670B-A40B: $H{=}7168$, $h_e{=}4096$, $N_e{=}128$, $N_a{=}4$, $TP{=}4$, $mbs{=}2$, $seq{=}4096$, $C{=}2$, BF16)

**G1 — EP via $M$** (headline). Here $M=64\,EP$, $n_{exp}=128/EP$, $n_{chunk}=n_{exp}/2$:

| EP  | $M$  | $n_{exp}$ | $n_{chunk}$ | $\eta$ (peak) | Regime |
| :-: | :--: | :-------: | :---------: | :-----------: | ------ |
|  8  | 512  |    16     |      8      |     0.23      | load-bound, deep pipe |
| 16  | 1024 |     8     |      4      |     0.47      | load-bound |
| 32  | 2048 |     4     |      2      |     0.93      | near-balanced but shallow pipe |

> The low activation ratio (3.125%) keeps $\eta<1$ until EP32; yet EP32 leaves only 2 chunks/rank,
> so the prologue erodes the device-level gain — and EP32 has the worst A2A (Microbenchmark A).
> This is the quantitative reason 670B-A40B offloading stays exposed-H2D-bound at small EP.

**G2 — Expert size $h_e \in \{2048, 4096, 8192\}$** (chunk map FFN2048→$C$4, 4096→$C$2, 8192→$C$1
to hold staging bytes ≈ const). Analytical $\eta$ flat; **measure the MFU/MBU climb** with the
larger GEMM/transfer.

**G3 — Hidden size $H \in \{4096, 7168, 8192\}$.** Absolute $T$ and the MBU-vs-MFU balance
(cf. H2D doc: $H{=}7168$ exposes more H2D than $H{=}4096$ at equal FFN).

**G4 — Chunk size $C \in \{1,2,4,8\}$.** Larger $C$ ↑ H2D MBU but ↑ staging memory
$2C\cdot 3Hh_e\,b$ and ↓ $n_{chunk}$ (coarser pipe). Locate the knee.

**G5 — Precision** BF16 vs FP8 (+ the TE quantize cost in the realistic path).

### Metrics — extend the H2D doc's 3-table layout

Reuse **Table 1** (H2D: $T_\text{load}$, MBU), **Table 2** (GEMM: $T_\text{gemm}$ FC1/FC2, TFLOPS,
MFU), **Table 3** (per-chunk overlap: exposed H2D, $\eta$); **add Table 4 — per-device pipeline**:
$n_{chunk}$, $T_\text{dev}$, device exposed H2D, device overlap efficiency, swept over EP{8,16,32}.

### Hypotheses

- **HB1**: per-chunk $\eta$ is **linear in $M$** (∴ in EP), crossing 1 near $M\approx1465$ at
  realized MBU/MFU.
- **HB2**: device overlap efficiency is **saturating / non-monotonic in EP** — it rises with $\eta$
  but is capped by shrinking $n_{chunk}$; expect a knee where more EP stops helping the pipeline.
- **HB3**: analytical $\eta$ is **flat in $h_e,H,C$** (G2–G4); any measured shift is realized
  MFU/MBU, not the ratio.
- **HB4**: per-rank exposed H2D falls $\sim 1/EP$ (less weight/rank + higher $\eta$) — the mirror
  image of A2A rising with EP in Microbenchmark A.

### Pitfalls

- **Shallow pipelines at large EP**: with $n_{chunk}=2$–4, fill/drain dominates — always report
  device-level, not just per-chunk, numbers.
- **Staging memory**: the double buffer is $2C\cdot 3Hh_e\,b$; large $C$/$h_e$ can OOM the staging
  area — record it per config.
- **Stream concurrency**: copy and compute streams must truly run in parallel — verify in Nsys that
  H2D and GEMM kernels overlap, and that NUMA binding (`--membind=0`) holds the ~450 GB/s.
- **Misleading $\eta$**: a high overlap efficiency can come from *low MFU* (a slow GEMM is easy to
  hide). Track MFU **alongside** $\eta$; the real target is low exposed H2D **at** high MFU.
- **Backward pass**: this models forward only; dgrad+wgrad add GEMMs and re-touch weights. Note
  whether `--delay-wgrad-compute` changes the load schedule before extending the model.

### Pseudo-code skeleton (device pipeline)

```python
def bench_device_pipeline(n_exp, C, M, H, he, dtype, iters=50, warmup=10):
    n = (n_exp + C - 1) // C
    cpu_w  = [pin_cpu(C, H, he, dtype) for _ in range(n)]      # weights in pinned host RAM
    gpu_w  = [empty_gpu(C, H, he, dtype) for _ in range(2)]    # double buffer
    copy_s, comp_s = torch.cuda.Stream(), torch.cuda.Stream()
    x = torch.randn(C, M, H, device="cuda", dtype=dtype)
    def run():
        with torch.cuda.stream(copy_s):
            gpu_w[0].copy_(cpu_w[0], non_blocking=True)         # prologue load
        for i in range(n):
            copy_s.synchronize()                               # wait load i
            if i + 1 < n:
                with torch.cuda.stream(copy_s):
                    gpu_w[(i+1) % 2].copy_(cpu_w[i+1], non_blocking=True)
            with torch.cuda.stream(comp_s):
                grouped_gemm(x, gpu_w[i % 2])                   # FC1 → SwiGLU → FC2
        comp_s.synchronize()
    for _ in range(warmup): run()
    # time `run` with CUDA events; also time load-only and gemm-only for Tables 1/2/3
# sweep: for EP in G1 (→ M, n_exp) / for he in G2 / for H in G3 / for C in G4 / dtype in G5
```

## Roadmap

1. **Microbenchmark A** — A2A dispatch/combine latency vs EP (S1–S5). *Designed above.*
2. **Microbenchmark B** — GroupedGEMM × H2D overlap per device vs EP/$h_e$/$H$/$C$ (G1–G5).
   *Designed above.*
3. **Combined EP-overlap run**: A2A on a side stream concurrent with GroupedGEMM (and, with
   offloading, the H2D pipeline of B) — the end-to-end overlap that A bounds from the communication
   side and B from the compute/H2D side. Report in the H2D doc's 3-table layout
   (isolated A2A / isolated GEMM / overlapped), extended with the device-pipeline Table 4.
