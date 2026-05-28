# MoE Pre-training Performance

## Preliminary

We need a target for the performance we expect to train a large MoE model. 

Assume we are going to train a 670B MoE model, and according to data team, we have roughly 15T - 20T tokens for pre-training. Let's take 17.5T as the mean value:

Given 4096 GPUs and 3 months:

$\rm{TokenThroughput} = \frac{17.5e12}{4096 \cdot 90 \cdot 24 \cdot 3600} = 550\ tokens/s/gpu$

This is the expected average token thourghput in real training, and for optimal balance throughput with `--force-load-balancing`:

$\rm{TokenThroughput'} = \frac{550}{0.8} = 687\ tokens/s/gpu$, where 0.8 is a penalty factor to compensate for the gap between real training throughput and balanced throughput in test. This value is observed in 30B-A3B training.

Hence, we need **at least 687 tokens/s/gpu** in the performance tests for a 670B MoE model. 

## General Setup

```
--transformer-impl transformer_engine
--main-grads-dtype fp32

# EP Overlap
--moe-token-dispatcher-type "alltoall"
--overlap-moe-expert-parallel-comm
--delay-wgrad-compute

--moe-grouped-gemm
--moe-permute-fusion
--moe-router-fusion
--force-load-balancing

--recompute-modules layernorm moe_act [mla_up_proj]

# if adam
--use-distributed-optimizer
--overlap-grad-reduce
--overlap-param-gather
```

**NOTE:** 

To guarantee that we align with the real training setup

- Don't use bf16 for main-grads
- Don't use precision-aware-optimizer
- Don't use capacity factor

## MoE-117B-A11B-0

**Mode Config**

```yaml
MODEL_NAME="moe_117b_a11b_0"

# general config
NUM_LAYERS=13
HIDDEN_SIZE=7168
FFN_HIDDEN_SIZE=14336
ATTENTION='gqa'
NUM_ATTENTION_HEADS=32
NUM_QUERY_GROUPS=8

# moe layer config
MOE_LAYER_FREQ='\([0]*3+[1]*10\)'
MOE_FFN_HIDDEN_SIZE=4096
MOE_SHARED_FFN_HIDDEN_SIZE=4096
NUM_EXPERTS=128
TOPK=8
```

**Setup**

- **Baseline**
  - `VPP_LAYOUT="Et\\|\\(tt\\|\\)*6,L"`
    - `--recompute-modules layernorm moe_act`
  
  - MBS = 2, GBS = 1024, EP16-TP4-PP4, GPUs = 64
  - MBS = 2, GBS = 2048, EP16-TP4-PP4, GPUs = 128
  
- **Offloading**
  - `VPP_LAYOUT="Et\\|\\(tt\\|\\)*6,L"`
    - `--recompute-modules layernorm moe_act`

  - MBS = 2, GBS = 1024, EP8-TP4-PP4, GPUs = 64
  - MBS = 2, GBS = 2048, EP8-TP4-PP4, GPUs = 128


**Results with 64 GPUs**

|                                                     | **Throughput (tokens/s/gpu)** | **Memory*** |  **MFU**  |
| --------------------------------------------------- | :---------------------------: | :---------: | :-------: |
| **FP8 MoE + EP8-TP4 + EP Overlap + MoE Offloading** |           **2560**            |    77.5%    | **18.5%** |
| FP8 MoE + EP16-TP4 + EP Overlap + MoE Offloading    |             2150              |    63.0%    |   15.7%   |
| **BF16 + EP16-TP4 + EP Overlap**                    |           **2220**            |    72.4%    | **16.0%** |

> ***NOTE:** memory is reported for the GPU with the highest memory pressure

## MoE-344B-A37B-1

```yaml
MODEL_NAME="moe_344b_a37b_1"

# general config
NUM_LAYERS=61
HIDDEN_SIZE=7168
FFN_HIDDEN_SIZE=16384
ATTENTION='mla'
NUM_ATTENTION_HEADS=128
NUM_QUERY_GROUPS=128

# moe layer config
MOE_LAYER_FREQ='\([0]*3+[1]*58\)'
MOE_FFN_HIDDEN_SIZE=4096
MOE_SHARED_FFN_HIDDEN_SIZE=2048
NUM_EXPERTS=64
TOPK=4
```

**Setup**

- **Baseline**

  - `VPP_LAYOUT="Et\\|\\(tt\\|\\)*30,L"`
    - `--recompute-modules layernorm moe_act mla_up_proj`
  - MBS = 2, GBS = 2048, EP8-TP4-PP16, GPUs = 512

- **Offloading**

  - `VPP_LAYOUT="Et\\|\\(tt\\|\\)*30,L"`

    1. `--recompute-modules layernorm moe_act mla_up_proj`

    2. `--recompute-modules layernorm moe_act`

  - MBS = 2, GBS = 2048, EP8-TP4-PP4, GPUs = 512

**Results with 512 GPUs**

|                                                   | **Throughput (tokens/s/gpu)** | TFLOP/s/GPU | **Memory** |  **MFU**  |
| ------------------------------------------------- | :---------------------------: | :---------: | :--------: | :-------: |
| FP8 MoE + EP4-TP4 + MoE Offloading-1              |              700              |     202     |     -      |   20.4%   |
| FP8 MoE + EP8-TP4 + EP Overlap + MoE Offloading-1 |            **765**            |   **221**   |   59.5%    | **22.5%** |
| FP8 MoE + EP8-TP4 + EP Overlap + MoE Offloading-2 |              800              |     231     |   80.0%    |   23.3%   |
| BF16 + EP8-TP4 + EP Overlap                       |                               |             |            |           |
| FP8 + EP4-TP4*                                    |            **879**            |    218.9    |     -      | **22.1%** |
| FP8 + EP8-TP4 + EP Overlap                        |                               |             |            |           |

> ***NOTE**: It is with BF16 main-grad + precision aware optimizer

## MoE-670B-A40B

```yaml
MODEL_NAME="moe_670b_a40b"

# general config
NUM_LAYERS=61
HIDDEN_SIZE=7168
FFN_HIDDEN_SIZE=16384
ATTENTION='mla'
NUM_ATTENTION_HEADS=128
NUM_QUERY_GROUPS=128

# moe layer config
MOE_LAYER_FREQ='\([0]*3+[1]*58\)'
MOE_FFN_HIDDEN_SIZE=4096
MOE_SHARED_FFN_HIDDEN_SIZE=2048
NUM_EXPERTS=128
TOPK=4
```

**Setup**

- **Baseline**
  - `VPP_LAYOUT="Et\\|\\(tt\\|\\)*30,L"`
    - `--recompute-modules layernorm moe_act mla_up_proj`
  - MBS = 2, GBS = 4096, GPUs = 512
- **Offloading**
  - `VPP_LAYOUT="Et\\|\\(tt\\|\\)*30,L"`
    - `--recompute-modules layernorm moe_act mla_up_proj`
  - MBS = 2, GBS = 4096, GPUs = 512

**Results with 512 GPUs**

|                                                        | **Throughput (tokens/s/gpu)** | TFLOP/s/GPU | **Memory** |  **MFU**  |
| ------------------------------------------------------ | :---------------------------: | :---------: | :--------: | :-------: |
| FP8 MoE + EP8-TP4 + EP Overlap + MoE Offloading        |              694              |     189     |   67.8%    |   18.7%   |
| FP8 MoE* + EP16-TP4 + EP Overlap + MoE Offloading      |            **772**            |   **200**   | **67.0%**  | **20.0%** |
| FP8 MoE + EP16-TP4 + EP Overlap + MoE Offloading$^{1}$ |              785              |     202     |   74.5%    |   20.4%   |
| BF16 + EP8-TP4 + EP Overlap                            |               -               |      -      |    OOM     |     -     |
| BF16 + EP16-TP4 + EP Overlap                           |               -               |      -      |    OOM     |     -     |
| BF16 + EP32-TP4 + EP Overlap                           |              627              |     152     |   81.2%    |   15.4%   |
| FP8* + EP16-TP4 + EP Overlap                           |               -               |      -      |    OOM     |     -     |
| FP8 + EP32-TP4 + EP Overlap                            |              670              |     168     |   80.0%    |   17.0%   |

> $^1$: this setup disable an activation recomputation in MoE layer.
>
> *NOTE: FP8 MoE only applies FP8 on MoE layer with offloading support.
>
> *NOTE: Transformer Engine FP8 implementations does not save memory consumption for some reason.
