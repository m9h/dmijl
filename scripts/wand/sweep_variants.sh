#!/bin/bash
#SBATCH --job-name=dfield-sweep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=1-8
#SBATCH --output=/data/datasets/wand/slurm/logs/sweep_%a.out
#SBATCH --error=/data/datasets/wand/slurm/logs/sweep_%a.err

echo "Experiment ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
julia --threads=2 /data/datasets/wand/slurm/sweep_variants.jl ${SLURM_ARRAY_TASK_ID}
echo "Done — $(date)"
