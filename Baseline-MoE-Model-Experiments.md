# Baseline MoE Model Experiments

> **Megatron commit**: [https://github.com/swiss-ai/Megatron-LM/tree/merge-260109](https://github.com/swiss-ai/Megatron-LM/tree/merge-260109)
>
> **Datasets**: `/iopsstor/scratch/cscs/anowak/datasets/megatron/llama_tokenized/fineweb-edu-100B/fineweb-edu-100B_00002_tokens`
>
> **Image**: `/iopsstor/scratch/cscs/gfu/ce-images/megatron_deepep-aarch64.sqsh`

## 03/03/2026 Draft model experiment

A draft model to make the experiment running and check:

- [x]  deepep
- model config
  
    ```json
    {
      "head_dim": 128,
      "hidden_size": 2048,
      "intermediate_size": 6144,
      "moe_intermediate_size": 768,
      "num_hidden_layers": 5,
      "num_attention_heads": 32,
      "num_key_value_heads": 4,
      "num_experts": 64,
      "num_experts_per_tok": 8,
      "act": "swiglu"
    }
    ```
    

## 05/03/2026 Reference Baseline MoE Model experiment

### Model

- **Qwen3-30B-A3B**
  
    ```json
    {
      "attention_bias": false,
      "attention_dropout": 0.0,
      **"head_dim": 128,**
      **"hidden_size": 2048,**
      "initializer_range": 0.02,
      **"intermediate_size": 6144,**
      "max_position_embeddings": 40960,
      **"moe_intermediate_size": 768,**
      "norm_topk_prob": true,
      **"num_hidden_layers": 48,**
      **"num_attention_heads": 32,
      "num_key_value_heads": 4,**
      **"num_experts": 128,**
      **"num_experts_per_tok": 8,
      "optimizer": "adam", # ademamix
      "act": "swiglu"**
    }
    ```
    
    - traditional GQA (4/32)
    - standard MoE sparsity (8/128)
    - reasonable model size

### Setup and Scripts

- **Megatron Recipe**: [link](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/qwen.html#id6)
- **Num of Node: 2**
- **Parallel Strategy and Estimated Memory per GPU**
    - PP4-EP2-DP2-TP1
    - GBS = 96
    - MBS = 4
    - GA = 12
    - SeqLen = 4096
    - Theoretical = 99150.76 MB (99.1GB)
        - Weight + optimizer = 44.18GB
        - Activation = 54.97GB
        - MoE activation recompute
        
        ```
        6: [Rank 6] (after 2 iterations) memory (MB) | allocated: 53199.095703125 | **max allocated: 84019.3681640625** | reserved: 84732.0 | max reserved: 84732.0
        2: [Rank 2] (after 2 iterations) memory (MB) | allocated: 50314.07666015625 | **max allocated: 81835.0830078125** | reserved: 82324.0 | max reserved: 82324.0
        4: [Rank 4] (after 2 iterations) memory (MB) | allocated: 50317.25390625 | **max allocated: 71306.03662109375** | reserved: 71744.0 | max reserved: 71744.0
        ```
    
- **Sbatch Script**: `/users/gfu/frameworks/Megatron-LM-sai/myscripts/apertus_qwen_30b_a3b_baseline.sh`

### Experiments

TBU