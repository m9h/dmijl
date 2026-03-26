#!/bin/bash
#SBATCH --job-name=axcal-map
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=1-10
#SBATCH --output=/data/datasets/wand/slurm/logs/axcal_map_%a.out
#SBATCH --error=/data/datasets/wand/slurm/logs/axcal_map_%a.err

# Split 4905 voxels across 10 Slurm array jobs (~490 each)
TOTAL=4905
PER_JOB=$(( (TOTAL + 9) / 10 ))
START=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_JOB + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * PER_JOB ))
if [ $END -gt $TOTAL ]; then END=$TOTAL; fi

echo "Job ${SLURM_ARRAY_TASK_ID}: voxels ${START}-${END} ($(date))"
julia --threads=2 /home/mhough/dev/dmijl/scripts/wand/axcaliber_slice_map_v2.jl 2000 ${START} ${END}
echo "Done ($(date))"
