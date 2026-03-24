#!/bin/bash
#SBATCH --job-name=surrogate_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/surrogate_train_%j.out
#SBATCH --error=slurm/logs/surrogate_train_%j.err

# ==========================================================================
# Surrogate network training on merged MCMRSimulator dataset.
#
# Trains a neural surrogate that maps microstructure parameters to dMRI
# signals, replacing expensive Monte Carlo forward simulations with a
# fast MLP inference (~1000x speedup).
#
# Architecture: h256_d6 (256-wide, 6-layer MLP with GELU + sigmoid output)
# Training: 100k steps, log-cosh loss, Adam optimizer
#
# Usage:
#   sbatch slurm/train_surrogate.sh
#
# Environment variables (optional overrides):
#   SURR_DATA_FILE    — path to merged JLD2 dataset
#                       (default: data/mcmr_training_data.jld2)
#   SURR_N_STEPS      — training steps (default: 100000)
#   SURR_HIDDEN_DIM   — hidden layer width (default: 256)
#   SURR_DEPTH        — network depth (default: 6)
#   SURR_BATCH_SIZE   — batch size (default: 512)
#   SURR_LR           — learning rate (default: 1e-3)
#   SURR_LOSS_TYPE    — loss function: mse, log_cosh, relative_mse
#                       (default: log_cosh)
#   SURR_CHECKPOINT   — output checkpoint path
#                       (default: checkpoints/surrogate.jld2)
# ==========================================================================

set -euo pipefail

echo "=========================================="
echo "Surrogate network training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# --- Configuration ---
DATA_FILE=${SURR_DATA_FILE:-data/mcmr_training_data.jld2}
N_STEPS=${SURR_N_STEPS:-100000}
HIDDEN_DIM=${SURR_HIDDEN_DIM:-256}
DEPTH=${SURR_DEPTH:-6}
BATCH_SIZE=${SURR_BATCH_SIZE:-512}
LR=${SURR_LR:-1e-3}
LOSS_TYPE=${SURR_LOSS_TYPE:-log_cosh}
CHECKPOINT=${SURR_CHECKPOINT:-checkpoints/surrogate.jld2}
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}

echo "  Data: $DATA_FILE"
echo "  Architecture: h${HIDDEN_DIM}_d${DEPTH}"
echo "  Steps: $N_STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Loss: $LOSS_TYPE"
echo "  Checkpoint: $CHECKPOINT"

# --- Load Julia and report GPU ---
if command -v module &>/dev/null; then
    module load julia/1.11 2>/dev/null || true
    module load cuda 2>/dev/null || true
fi

julia --version
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "No nvidia-smi available"
echo ""

# --- Run training ---
julia --project="$PROJECT_DIR" --threads=4 -e "
using Random, Statistics, Printf
using JLD2
using Lux, Optimisers, Zygote
using CUDA, LuxCUDA

# Include source files
const PROJECT_ROOT = \"$PROJECT_DIR\"
include(joinpath(PROJECT_ROOT, \"src/noise.jl\"))
include(joinpath(PROJECT_ROOT, \"src/diffusion/schedule.jl\"))
include(joinpath(PROJECT_ROOT, \"src/pinn/bloch_torrey.jl\"))

# GPU setup
const USE_GPU = CUDA.functional()
const dev = USE_GPU ? gpu_device() : cpu_device()
println(\"GPU available: \$USE_GPU\")
if USE_GPU
    println(\"GPU: \", CUDA.name(CUDA.device()))
end

# Configuration
const N_STEPS    = $N_STEPS
const HIDDEN_DIM = $HIDDEN_DIM
const DEPTH      = $DEPTH
const BATCH_SIZE = $BATCH_SIZE
const LR         = $LR
const LOSS_TYPE  = :$LOSS_TYPE

# Load merged training data
data_path = joinpath(PROJECT_ROOT, \"$DATA_FILE\")
println(\"Loading training data from \$data_path ...\")
data = load(data_path)
all_params  = data[\"all_params\"]   # (param_dim, n_samples)
all_signals = data[\"all_signals\"]  # (signal_dim, n_samples)

param_dim  = size(all_params, 1)
signal_dim = size(all_signals, 1)
n_total    = size(all_params, 2)

@printf(\"  Dataset: %d samples, param_dim=%d, signal_dim=%d\\n\",
        n_total, param_dim, signal_dim)

# Normalise parameters to [0, 1] for training
# (MCMRSimulator params are already in physical units; we min-max normalise)
param_mins = minimum(all_params, dims=2)
param_maxs = maximum(all_params, dims=2)
param_spans = max.(param_maxs .- param_mins, 1f-8)
params_norm = (all_params .- param_mins) ./ param_spans

# b0-normalise signals
b0_mask = data[\"bval_list\"] .< 0.01
for j in 1:n_total
    b0_mean = mean(all_signals[b0_mask, j])
    b0_mean = max(b0_mean, 1f-6)
    all_signals[:, j] ./= b0_mean
end

println(\"  Signal range after b0-norm: \", extrema(all_signals))

# Build surrogate model
rng = MersenneTwister(42)
surrogate = build_surrogate(
    param_dim  = param_dim,
    signal_dim = signal_dim,
    hidden_dim = HIDDEN_DIM,
    depth      = DEPTH,
)
ps, st = Lux.setup(rng, surrogate)

# Move to GPU
if USE_GPU
    ps = ps |> dev
    st = st |> dev
end

# Data sampling function (draw random minibatches from precomputed dataset)
function data_fn(rng, n)
    idx = rand(rng, 1:n_total, n)
    p = params_norm[:, idx]
    s = all_signals[:, idx]
    if USE_GPU
        return dev(p), dev(s)
    end
    return p, s
end

# Train
println(\"\\nTraining surrogate: h\$(HIDDEN_DIM)_d\$(DEPTH), \$N_STEPS steps\")
t0 = time()
ps, st, losses = train_surrogate!(
    surrogate, ps, st, data_fn;
    n_steps       = N_STEPS,
    batch_size    = BATCH_SIZE,
    learning_rate = LR,
    print_every   = max(1, N_STEPS ÷ 20),
    loss_type     = LOSS_TYPE,
)
train_time = time() - t0
@printf(\"Training complete: %.1fs (%.0f steps/s)\\n\", train_time, N_STEPS / train_time)

# Move back to CPU for evaluation and saving
ps_cpu = USE_GPU ? (ps |> cpu_device()) : ps
st_cpu = USE_GPU ? (st |> cpu_device()) : st

# Evaluate on held-out test split
n_test = min(1000, n_total ÷ 10)
test_idx = rand(MersenneTwister(999), 1:n_total, n_test)
pred_signals, _ = surrogate(params_norm[:, test_idx], ps_cpu, st_cpu)
test_signals = all_signals[:, test_idx]

rel_errors = Float64[]
for j in 1:n_test
    pred = pred_signals[:, j]
    exact = test_signals[:, j]
    mask = exact .> 0.01
    if any(mask)
        re = mean(abs.(pred[mask] .- exact[mask]) ./ exact[mask])
        push!(rel_errors, re)
    end
end

@printf(\"\\nEvaluation on %d test samples:\\n\", n_test)
@printf(\"  Median relative error: %.4f (%.2f%%)\\n\", median(rel_errors), median(rel_errors)*100)
@printf(\"  Samples <1%% error:    %.1f%%\\n\", count(rel_errors .< 0.01) / length(rel_errors) * 100)
@printf(\"  Samples <5%% error:    %.1f%%\\n\", count(rel_errors .< 0.05) / length(rel_errors) * 100)

# Save checkpoint
ckpt_path = joinpath(PROJECT_ROOT, \"$CHECKPOINT\")
mkpath(dirname(ckpt_path))
@save ckpt_path ps_cpu st_cpu losses param_mins param_maxs param_spans train_time HIDDEN_DIM DEPTH LOSS_TYPE N_STEPS

println(\"\\nCheckpoint saved: \$ckpt_path\")
"

echo ""
echo "End: $(date)"
