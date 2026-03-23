#!/bin/bash
#SBATCH --job-name=surrogate_sweep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=1-12
#SBATCH --output=slurm/logs/surrogate_%a.out
#SBATCH --error=slurm/logs/surrogate_%a.err

echo "Running experiment ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Start: $(date)"

julia --threads=2 slurm/surrogate_sweep.jl ${SLURM_ARRAY_TASK_ID}

echo "End: $(date)"
