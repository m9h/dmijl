#!/bin/bash
#SBATCH --job-name=dfield-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/datasets/wand/slurm/logs/dfield_cpu_%j.out
#SBATCH --error=/data/datasets/wand/slurm/logs/dfield_cpu_%j.err

echo "=== D(r) Recovery: CPU ==="
echo "Host: $(hostname), Start: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

julia --threads=$SLURM_CPUS_PER_TASK /data/datasets/wand/slurm/run_dfield.jl cpu

echo "End: $(date)"
