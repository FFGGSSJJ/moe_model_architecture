# MoE Pre-training Performance

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

|                                                       | **Throughput (tokens/s/gpu)** | TFLOP/s/GPU | **Memory** |  **MFU**  |
| ----------------------------------------------------- | :---------------------------: | :---------: | :--------: | :-------: |
| FP8 MoE + EP4-TP4 + MoE Offloading-1                  |              700              |     202     |     -      |   20.4%   |
| **FP8 MoE + EP8-TP4 + EP Overlap + MoE Offloading-1** |            **765**            |   **221**   |   59.5%    | **22.5%** |
| FP8 MoE + EP8-TP4 + EP Overlap + MoE Offloading-2     |              800              |     231     |   80.0%    |   23.3%   |
| **FP8 + EP4-TP4***                                    |            **879**            |    218.9    |     -      | **22.1%** |

> ***NOTE**: It is with BF16 main-grad + precision aware optimizer, and the numbers have to be reverified

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

|                                                      | **Throughput (tokens/s/gpu)** | TFLOP/s/GPU | **Memory** |  **MFU**  |
| ---------------------------------------------------- | :---------------------------: | :---------: | :--------: | :-------: |
| FP8 MoE + EP8-TP4 + EP Overlap + MoE Offloading      |              694              |     200     |   67.8%    |   20.2%   |
| **FP8 MoE + EP16-TP4 + EP Overlap + MoE Offloading** |            **712**            |   **205**   | **67.0%**  | **20.8%** |
| BF16 + EP8-TP4 + EP Overlap                          |               -               |      -      |    OOM     |     -     |
| BF16 + EP16-TP4 + EP Overlap                         |               -               |      -      |    OOM     |     -     |
| FP8 + EP16-TP4 + EP Overlap                          |                               |             |            |           |