#!/bin/bash
#SBATCH --job-name=fp8_a2a_internode
#SBATCH --output=fp8_a2a_internode_%j.out
#SBATCH --error=fp8_a2a_internode_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --partition=normal
#SBATCH --account=infra01
#
# ==============================================================================
# FP8 All-to-All Inter-Node Test & Benchmark
#
# Usage:
#   sbatch run_fp8_a2a_internode.sbatch                      # defaults (2 nodes × 4 GPUs)
#   sbatch -N4 --ntasks-per-node=8 run_fp8_a2a_internode.sbatch  # 4 nodes × 8 GPUs
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Nodes        : $SLURM_JOB_NUM_NODES"
echo "Tasks/node   : $SLURM_NTASKS_PER_NODE"
echo "Total GPUs   : $(( SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE ))"
echo "Node list    : $SLURM_JOB_NODELIST"
echo "Start time   : $(date)"
echo "======================================================================"

# ---- Distributed environment setup ----
export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=1

# NCCL tuning for multi-node (adjust for your interconnect)
export NCCL_DEBUG=INFO          # set to WARN for production benchmarks
export NCCL_IB_DISABLE=0        # enable InfiniBand if available
export NCCL_IB_HCA=mlx5        # adjust to match your IB device names
export NCCL_NET_GDR_LEVEL=5     # enable GPU-Direct RDMA

echo "MASTER_ADDR  : $MASTER_ADDR"
echo "MASTER_PORT  : $MASTER_PORT"

# ---- Run the test ----
SCRIPT_DIR=/capstor/scratch/cscs/gfu/frameworks/moe_model_architecture/moe-pretraining-perf/tests

export WORLD_SIZE=$SLURM_NPROCS

srun --environment=pytorch_env \
     --mpi=pmix \
     --distribution=block:block \
     --network=disable_rdzv_get \
     --ntasks="$(( SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE ))" \
     --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
     -lu bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID python -u $SCRIPT_DIR/fp8_a2a_internode_test.py"

echo "======================================================================"
echo "End time     : $(date)"
echo "======================================================================"
