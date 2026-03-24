#!/bin/bash
#SBATCH --job-name=mcmr_datagen
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=1-50
#SBATCH --output=slurm/logs/mcmr_datagen_%a.out
#SBATCH --error=slurm/logs/mcmr_datagen_%a.err

# ==========================================================================
# MCMRSimulator data generation — job array
#
# Generates (params, signals) training pairs by running Monte Carlo random
# walks through packed-cylinder geometries via MCMRSimulator.jl.
#
# 50,000 total configurations split across 50 array tasks (1,000 per task).
# Each configuration runs 50,000-100,000 spins (~30 seconds per config).
#
# Usage:
#   sbatch slurm/generate_mcmr_data.sh
#
# Environment variables (optional overrides):
#   MCMR_TOTAL_CONFIGS  — total configurations (default: 50000)
#   MCMR_N_SPINS        — spins per config (default: 50000)
#   MCMR_GEOMETRY       — geometry type: cylinders or spheres (default: cylinders)
#   MCMR_OUTPUT_DIR     — output directory (default: data/mcmr_chunks)
# ==========================================================================

set -euo pipefail

echo "=========================================="
echo "MCMRSimulator data generation"
echo "Array task: ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_COUNT:-50}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# --- Configuration ---
TOTAL_CONFIGS=${MCMR_TOTAL_CONFIGS:-50000}
N_SPINS=${MCMR_N_SPINS:-50000}
GEOMETRY=${MCMR_GEOMETRY:-cylinders}
OUTPUT_DIR=${MCMR_OUTPUT_DIR:-data/mcmr_chunks}
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
N_TASKS=${SLURM_ARRAY_TASK_COUNT:-50}

# Compute chunk size and range for this task
CHUNK_SIZE=$(( (TOTAL_CONFIGS + N_TASKS - 1) / N_TASKS ))
START_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK_SIZE + 1 ))
END_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
if [ "$END_IDX" -gt "$TOTAL_CONFIGS" ]; then
    END_IDX=$TOTAL_CONFIGS
fi
ACTUAL_CHUNK=$(( END_IDX - START_IDX + 1 ))

echo "  Total configs: $TOTAL_CONFIGS"
echo "  Spins per config: $N_SPINS"
echo "  Geometry: $GEOMETRY"
echo "  This task: configs $START_IDX-$END_IDX ($ACTUAL_CHUNK configs)"
echo "  Output: $OUTPUT_DIR"

# --- Load Julia (try module system first, fall back to juliaup/PATH) ---
if command -v module &>/dev/null; then
    module load julia/1.11 2>/dev/null || true
fi

julia --version
echo ""

# --- Run data generation ---
julia --project="$PROJECT_DIR" --threads=2 -e "
using Random, Statistics, Printf
using JLD2
using MCMRSimulator
using MRIBuilder

# Include source files
const PROJECT_ROOT = \"$PROJECT_DIR\"
include(joinpath(PROJECT_ROOT, \"src/pipeline/mcmr_generator.jl\"))

# Configuration from environment
const N_SAMPLES   = $ACTUAL_CHUNK
const N_SPINS     = $N_SPINS
const GEOMETRY    = :$GEOMETRY
const TASK_ID     = $SLURM_ARRAY_TASK_ID
const START_IDX   = $START_IDX
const OUTPUT_DIR  = joinpath(PROJECT_ROOT, \"$OUTPUT_DIR\")

# Seed reproducibly per task
rng = MersenneTwister(42 + TASK_ID)
Random.seed!(42 + TASK_ID)

# Define multi-shell acquisition (HCP-like)
bvals_shells = [0.0, 1.0, 2.0, 3.0]
n_per_shell  = [6, 30, 30, 24]  # full HCP: 90 measurements

function random_perp_directions(rng, n)
    angles = rand(rng, n) .* 2pi
    return [[cos(a), sin(a), 0.0] for a in angles]
end

sequences = Sequence[]
bval_list = Float64[]
for (bval, n_dir) in zip(bvals_shells, n_per_shell)
    if bval == 0.0
        for _ in 1:n_dir
            push!(sequences, DWI(bval=0.0, TE=80.0, scanner=Siemens_Prisma))
            push!(bval_list, 0.0)
        end
    else
        dirs = random_perp_directions(rng, n_dir)
        for d in dirs
            push!(sequences, DWI(
                bval=bval, TE=80.0, scanner=Siemens_Prisma,
                gradient=(orientation=d, rise_time=:min),
            ))
            push!(bval_list, bval)
        end
    end
end

const N_MEAS = length(sequences)
println(\"Acquisition: \$N_MEAS measurements across shells b=\$bvals_shells\")

# Select geometry sampler
sampler = GEOMETRY == :cylinders ? sample_cylinder_geometry : sample_sphere_geometry

# Generate training data
println(\"Generating \$N_SAMPLES configs with \$N_SPINS spins each...\")
params, signals = generate_mcmr_training_data(
    sampler, sequences, N_SAMPLES;
    n_spins     = N_SPINS,
    diffusivity = 2.0,
    R2          = 0.0,
    R1          = 0.0,
    verbose     = true,
)

# Save chunk
mkpath(OUTPUT_DIR)
chunk_file = joinpath(OUTPUT_DIR, \"chunk_\$(lpad(TASK_ID, 4, '0')).jld2\")
@save chunk_file params signals bval_list N_SPINS GEOMETRY START_IDX

println(\"Saved \$chunk_file\")
println(\"  params shape: \$(size(params))\")
println(\"  signals shape: \$(size(signals))\")
"

echo ""
echo "End: $(date)"
