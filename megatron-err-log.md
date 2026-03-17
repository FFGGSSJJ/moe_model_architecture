# Megatron Error Log

> - Megatron: https://github.com/swiss-ai/Megatron-LM/tree/fuguan/moe_arch

## 1. Checkpoint Loading Error: OOM

- **Problems:** 
  There are 2 problems
  1. When `--precision-aware-optimizer` is enabled, this version of Megatron will save Optimizer States with an additional bool field `padding`. However in`megatron/core/optimizer/distrib_optimizer.py`, the function `_set_main_param_and_optimizer_states` expects `tensors` to contains only 3 fields. But what the saved tensor has an extra field called `padding` which is a bool type, and it causes error in when calling `set_scaled_state` because `padding` is not a tensor.
  2. When loading optimizer states from checkpoints, the memory cost is unexpectedly large and could cause OOM.

- **Solutions:**
  1. Either use `dev` branch in Megatron, or skip `padding` field manually in `_set_main_param_and_optimizer_states`
  2. For OOM: https://github.com/NVIDIA/Megatron-LM/pull/3558

## 2. Checkpoint Saving Error

- **Problems**:
  1. `--precision-aware-optimizer` + `--dist-ckpt-optim-fully-reshardable` is not compatible. If checkpoints have to be fully reshardable, use full precision optimizer. Otherwise, there will be loading error when loading a fully reshardable checkpoint
  2. `--dist-ckpt-optim-fully-reshardable`  may cause OOM during checkpoints saving if the memory pressure is high. 
  3. `args.tokens_so_far` is not initialized in `training.py` if the training starts from scratch

## 3. Mysterious Error

- **Problems**:
  1. with `--moe-permute-fusion` enabled, the training from scratch will fail after 3 steps.