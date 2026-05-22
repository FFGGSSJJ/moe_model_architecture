# MoE Pre-training Performance

## General Setup

```
--transformer-impl transformer_engine
--main-grads-dtype fp32

--moe-token-dispatcher-type "alltoall"
--overlap-moe-expert-parallel-comm
--delay-wgrad-compute

--moe-grouped-gemm
--moe-permute-fusion
--moe-router-fusion
--force-load-balancing

--recompute-modules layernorm moe_act

# if adam
--use-distributed-optimizer
--overlap-grad-reduce
--overlap-param-gather
```

**NOTE:** 

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
  - MBS = 2, GBS = 1024, EP16-TP4-PP4, GPUs = 64
  - MBS = 2, GBS = 2048, EP16-TP4-PP4, GPUs = 128
  
- **Offloading**
  - `VPP_LAYOUT="Et\\|\\(tt\\|\\)*6,L"`
  - MBS = 2, GBS = 1024, EP8-TP4-PP4, GPUs = 64
  - MBS = 2, GBS = 2048, EP8-TP4-PP4, GPUs = 128


**Results with 64 GPUs**

|                                                     | **Throughput (tokens/s/gpu)** | **Memory** |  **MFU**  |
| --------------------------------------------------- | :---------------------------: | :--------: | :-------: |
| **FP8 MoE + EP8-TP4 + EP Overlap + MoE Offloading** |           **2560**            |   77.5%    | **18.5%** |
| FP8 MoE + EP16-TP4 + EP Overlap + MoE Offloading    |             2150              |   63.0%    |   15.7%   |
| **BF16 + EP16-TP4 + EP Overlap**                    |           **2220**            |   72.4%    | **16.0%** |