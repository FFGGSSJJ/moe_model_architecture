# Offload Experts in Megatron

The GH200 cluster we have suffers from low All-to-All bandwidth, which makes inter-node EP communication impractical. If we want to train a large MoE model efficiently at large scale, we have to figure out a way to fill the model into GPUs under EP=4 constraint.

GH200 has 450GB/s H2D bandwidth, which makes it possible to offload experts into CPU RAM while hide the H2D loading with computation. 

## Theoratical Analysis

We define the following symbols for MoE model:

| Symbol | Def                                        |
| ------ | ------------------------------------------ |
| $N_e$  | Total number of routed experts             |
| $N_a$  | Activated routed experts per token         |
| $H$    | Hidden size of model                       |
| $h_e$  | Intermediate size of expert                |
| $L$    | Hidden layers                              |
| $C$    | Chunk size (number of experts) for loading |

And the following for parallel strategy:

- $N_{gpu}$ = world size
- DP = $N_{gpu}$ / (TP * PP)
- EDP = $N_{gpu}$ / (EP * PP)

#### GPU RAM Analysis for MoE Layer (TBD)

This part of analysis should provide guidance on the model size we are going to have. 

- ${\rm{Mem}}_{e} = N_e \cdot H h_e\cdot 3 \cdot L / \rm{EP} \cdot 2Byte$
- ${\rm{Mem}}_{grad} = N_e \cdot H h_e\cdot 3 \cdot L / \rm{EP} \cdot 4\rm{Byte}$
- ${\rm{Mem}}_{adam} = (N_e \cdot H h_e\cdot 3 \cdot L) \cdot 3  / \rm{EP}/ \rm{EDP} \cdot 4\rm{Byte}$

Megatron by default uses ZeRO-1 to shard optimizer states. Hence at large scale, main gradient takes the majority of GPU RAM. If we offload all experts into CPU RAM, then we are bounded by gradient size and activation memory.

#### Loading-Computation Overlap Efficiency

This part of analysis should provide guidance on parallel strategy and MoE model configuration.

<img src="./figs/offloading/pipeline.png" alt="exploss2" style="zoom:50%;" />

Let $C$ be the chunk size that indicates number of experts to load at a time and $M$ be the number of tokens received by each expert.

- $T_{load} = \frac{C \cdot H\cdot 2h_e \cdot 2Byte}{450GB/s}$, $T_{gemm} = \frac{C \cdot 2 \cdot M \cdot H\cdot 2h_e}{\rm{TFLOP/s}}$

- $\rm{Overlap Efficiency =}\frac{T_{gemm}}{T_{load}}$

To get **perfect overlap efficiency of 100%:**

- In the ideal scenario, we achieve 100% MBU and 100% MFU, we have: $M = \frac{989T}{450G} \approx 2200$ tokens/expert

- If we achieve 90% MBU and 60% MFU, we have: $M = \frac{0.6 \cdot 989T}{0.9 \cdot 450G} \approx 1465$ tokens/expert

  - Test with DeepSeek-V3 configuration:

  ``````
  [H2D] 16 experts, 0.47 GB total
    Shape per tensor: (7168, 4096), dtype: bfloat16
    FC1 Latency: 2.232 ms | Bandwidth: 421.00 GB/s | MBU: 93.56%
    FC2 Latency: 1.119 ms | Bandwidth: 419.85 GB/s | MBU: 93.30%
  [GroupedGEMM FWD] 16 experts, 1465 tokens/expert, (7168, 4096)
    Shape of A: (23440, 7168), Shape of B: (16, 7168, 4096), 
    FC1 Latency: 2.263 ms | TFLOPS: 608.30 | MFU: 61.51%
    FC2 Latency: 1.285 ms | TFLOPS: 535.66 | MFU: 54.16%
  
  Exposed H2D Latency: FC1=-0.031 ms | FC2=-0.166 ms
  Overlap efficiency: FC1=101.39% | FC2=114.83%
  ``````

We need to get **$M$ <u>at least larger than 1000 tokens</u>**. Larger $M$ means better overlapping efficiency. We define $M$ in a balanced scenario as:

- $k = \frac{DP}{EDP} = \frac{EP}{TP}$, $R_a = \frac{N_a}{N_e}$
- $M = \frac{mbs\cdot seq \cdot N_a}{N_e} \cdot \frac{DP}{EDP} = mbs\cdot seq \cdot \frac{EP}{TP}\cdot R_a$ 

We assume $mbs=2$ and $seq=4096$, this makes the balance token number only relate to EP, TP and activated ratio. 

- If TP=EP=4, $M = 8192 \cdot R_a \Rightarrow R_a \geq 12$%

- if TP=2, EP=4, $M = 8192 \cdot 2 \cdot R_a \Rightarrow R_a \geq 6$%

The above discussion is built on the assumption that we need 100% overlap efficiency. 

## Implementation in Megatron

Integration in Megatron is tricky, and takes time for finetuning performance. The roadmap of development:

1. Expert Weight Offloading
2. Master Weight Offloading
3. Other Optimizer States Offloading

We keep main gradient on the GPU RAM to avoid frequent H2D and D2H transfer during bakcward pass.

### Step1: Expert Weight Offloading

> - Megatron: https://github.com/FFGGSSJJ/Megatron-LM/tree/moe_arch/dev
> - groupgemm: https://github.com/FFGGSSJJ/grouped_gemm/tree/dev/grad_acc_fuse

The majority of changes happened in Megatron to support expert weight offloading in a non-aggressive way. To enable better control of computation and support low-overhead batched h2d copy, a separate groupgemm repo is used.

- Design Expert Layer autograd function
  - Loading-Computation Pipieline
    - https://github.com/FFGGSSJJ/Megatron-LM/tree/moe_arch/dev
  - Batched-H2D Copy for Lower CPU overhead.
    - https://github.com/FFGGSSJJ/grouped_gemm/tree/dev/grad_acc_fuse
- Modify `_ParamAndGradBuffer`: allocate CPU param_data buffer for expert parameters
- Modify `DistributedOptimizer`: for parameter with CPU storage, allocate FP32 GPU storage for master weight
- Modify `_ParamAndGradBucketGroup.start_param_sync()`: for CPU parameter, all-gather has to use NCCL. First copy CPU parameter to temporary GPU tensor, perform all-gather, and copy back to CPU tensor.

### Step2: Master Weight Offloading (TBD)

TBD

### Step3: Optimizer States Offloading (TBD)

TBD

## Verification of Implementation

There will be mainly 2 part of verification: correctness and performance. 

### Correctness Verification

On going.

### Performance Verification

As we have built previously, the overlapping efficiency is mainly related to the number of tokens each experts can receive in balanced scenario. We design 2 models:

#### 1. MoE-40B-A4B

- N_e = 448, N_a = 28, R_a = 6.25%, H = 4096, h_4 = 2048
- The purpose of this model is to saturate GPU RAM with TP4-EP4 + Expert Offloading on 32 GPUs. 
  - Overlap Efficiency ~= 40%
- We expect that **EP4 + Expert Offloading to outperform EP8.**

- Nsys Profile

<img src="./figs/offloading/offloading-pipeline-0422-01.png" alt="exploss2" style="zoom:50%;" />

- **Results**:

  - **EP4 + Offloading:** 8700 tokens/s/gpu, 92% GPU RAM
  - **EP8:** 7000 tokens/s/gpu

- Red for EP4-TP4+Offloading:

  <img src="./figs/offloading/moe-40b-a4b.png" alt="exploss2" style="zoom:50%;" />

#### 2. MoE-27B-A3B

- N_e = 256, N_a = 16, R_a = 6.25%, H = 4096, h_4 = 2048
- The purpose of this model is to make EP4-TP2 possible on 32 GPUs
  - Overlap Efficiency ~= 70%
- We expect that **EP4 + Expert Offloading has close performance to EP4.**

- **Results**:
  - **EP4 + Offloading:** 17000 tokens/s/gpu
  - **EP4:** 18100 tokens/s/gpu
- Red for EP4-TP2 + Offloading

<img src="./figs/offloading/moe-27b-a3b.png" alt="exploss2" style="zoom:50%;" />