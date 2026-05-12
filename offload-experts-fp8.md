# Offload Experts with FP8 support in Megatron

As we have built in offload-experts.md, it is verified to be practical offload experts into CPU RAM while hide the H2D loading with computation. 

In a next step, FP8 support should be included into the roadmap. 

## Theoratical Analysis

### FP8 Recipe

DeepSeek recipe:

| Model Weight Quantization | Activation Quantization | FP8 Dtype   | SF Dtype |
| ------------------------- | ----------------------- | ----------- | -------- |
| (128, 128)                | (1, 128)/(128, 1)       | float8_e4m3 | float32  |

### Loading-Computation Overlap Efficiency

The same analysis can be applied. With half model weights (FP8) and double computational TFLOPs (FP8), we should have the same conclusion on optimal token number $M$.

## Implementation in Megatron

FP8 implementations will be tricky. 

For now only TransformerEngine provides solution for FP8 computation in MoE and Attention layer, however to support offloading in MoE it will hard to hack TE code. It is a relatively easier approach to implement our own autograd function with FP8 support for MLP. The roadmap is:

1. **FP8 MoE**: `experts_fp8_util.py`

   It is the reference implementation without offloading.

   - **MLP:** `ExpertsFP8GroupedSwiMLP`
   - **Parameter:** `FP8GPUExpertsParameterManager`

2. **FP8 MoE with Expert Offloading**: `experts_offloading_fp8_util.py`

   - **MLP**: `OffloadingExpertsFP8GroupedSwiMLP`

     - **FP8 GroupGEMM**: DeepGEMM
       - Forward and Activation Backward: `m_grouped_fp8_gemm_nt_contiguous`

       - Weight Backward: `k_grouped_fp8_gemm_nt_contiguous`

     - **FP8 Quantizations**: triton kernels `fp8_jit.py`
       - per_block_cast (128, 128): for weight

       - per_token_cast (1, 128): for forward activation

       - per_channel_cast (128, 1): for backward activation

     - **Loading-Computation Pipieline**: same as `OffloadingExpertsGroupedSwiMLP`

   - **Parameter**: `FP8ExpertsParameterManager`

     Expert parameters are stored on CPU RAM. Megatron DDP manages the allocation of parameter tensor. Coupled with complex buckets and overlapping logic, it is not ideal to aggressively modify existing DDP codes. Hence, `FP8ExpertsParameterManager` is designed as an extra state manager to control the behavior of FP8 weights quantization and access.

     - **Allocated** as BF16 buffers.
       - Megatron DDP handles the allocation of BF16 Parameter on CPU
       - `FP8ExpertsParameterManager` uses `.data_ptr()` to record expert parameter tensor storage.
     - **Quantized** after all-gather or before first micro-batch computation in each iteration. These are 2 different approaches. The first is the ideal one to avoid prolonging PP bubble, but difficult in practice. For now `FP8ExpertsParameterManager` takes the second approach.
       1. Upon all-gather the BF16 tensors stay on GPU, and can be directly quantized into FP8 tensors before copying to CPU. Expert parameters are stored in the same bucket group, when the bucket group finishes param_sync, params in the group can be marked for quantization. 
       2. Upon the first micro-batch computation, all BF16 parameters have been updated by optimizer. When they are first accessed, re-quantize them.
          - `FP8ExpertsParameterManager` should be aware of `is_first_mircobatch` to mark when requantization should happen.
          - `FP8ExpertsParameterManager` will handle quantization through an unified interface:
            1. **H2D Copy**: copy updated CPU parameter into GPU buffer
            2. **Quantization**: perform quantization on GPU
            3. **D2H Copy**: **<u>reuse</u>** existing BF16 buffer to save quantized parameter

## Implementation Log