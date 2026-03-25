#!/bin/bash
#SBATCH --job-name=dfield-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/datasets/wand/slurm/logs/dfield_gpu_%j.out
#SBATCH --error=/data/datasets/wand/slurm/logs/dfield_gpu_%j.err

echo "=== D(r) Recovery: GPU ==="
echo "Host: $(hostname), Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

julia --threads=$SLURM_CPUS_PER_TASK /data/datasets/wand/slurm/run_dfield.jl gpu

echo "End: $(date)"
