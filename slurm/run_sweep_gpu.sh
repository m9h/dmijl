#!/bin/bash
#SBATCH --job-name=surrogate_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=1-12
#SBATCH --output=slurm/logs/surrogate_gpu_%a.out
#SBATCH --error=slurm/logs/surrogate_gpu_%a.err

echo "Running experiment ${SLURM_ARRAY_TASK_ID} on $(hostname) [GPU]"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

julia --threads=4 slurm/surrogate_sweep_gpu.jl ${SLURM_ARRAY_TASK_ID}

echo "End: $(date)"
