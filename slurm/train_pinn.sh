#!/bin/bash
#SBATCH --job-name=pinn_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/pinn_finetune_%j.out
#SBATCH --error=slurm/logs/pinn_finetune_%j.err

# ==========================================================================
# PINN fine-tuning — Bloch-Torrey PDE residual loss.
#
# Loads a pre-trained surrogate and fine-tunes it with a combined loss:
#   L = L_data + lambda * L_pde
#
# where L_data is supervised MSE from MCMRSimulator data and L_pde is the
# Bloch-Torrey PDE residual evaluated at random collocation points.
# This physics-informed refinement improves the surrogate's generalization
# to restricted diffusion regimes not well covered by the training data.
#
# Usage:
#   sbatch slurm/train_pinn.sh
#
# Environment variables (optional overrides):
#   PINN_SURR_CKPT     — pre-trained surrogate checkpoint
#                         (default: checkpoints/surrogate.jld2)
#   PINN_DATA_FILE     — merged training dataset
#                         (default: data/mcmr_training_data.jld2)
#   PINN_N_STEPS       — fine-tuning steps (default: 20000)
#   PINN_BATCH_SIZE    — supervised batch size (default: 256)
#   PINN_N_COLLOC      — collocation points per step (default: 128)
#   PINN_LR            — learning rate (default: 1e-4)
#   PINN_LAMBDA_PDE    — PDE loss weight (default: 0.1)
#   PINN_CHECKPOINT    — output checkpoint path
#                         (default: checkpoints/surrogate_pinn.jld2)
# ==========================================================================

set -euo pipefail

echo "=========================================="
echo "PINN fine-tuning (Bloch-Torrey residual)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# --- Configuration ---
SURR_CKPT=${PINN_SURR_CKPT:-checkpoints/surrogate.jld2}
DATA_FILE=${PINN_DATA_FILE:-data/mcmr_training_data.jld2}
N_STEPS=${PINN_N_STEPS:-20000}
BATCH_SIZE=${PINN_BATCH_SIZE:-256}
N_COLLOC=${PINN_N_COLLOC:-128}
LR=${PINN_LR:-1e-4}
LAMBDA_PDE=${PINN_LAMBDA_PDE:-0.1}
CHECKPOINT=${PINN_CHECKPOINT:-checkpoints/surrogate_pinn.jld2}
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}

echo "  Surrogate checkpoint: $SURR_CKPT"
echo "  Data file: $DATA_FILE"
echo "  Steps: $N_STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Collocation points: $N_COLLOC"
echo "  Learning rate: $LR"
echo "  Lambda PDE: $LAMBDA_PDE"
echo "  Checkpoint: $CHECKPOINT"

# --- Load Julia and report GPU ---
if command -v module &>/dev/null; then
    module load julia/1.11 2>/dev/null || true
    module load cuda 2>/dev/null || true
fi

julia --version
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "No nvidia-smi available"
echo ""

# --- Run PINN fine-tuning ---
julia --project="$PROJECT_DIR" --threads=4 -e "
using Random, Statistics, Printf, LinearAlgebra
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
const BATCH_SIZE = $BATCH_SIZE
const N_COLLOC   = $N_COLLOC
const LR         = $LR
const LAMBDA_PDE = $LAMBDA_PDE

# Load pre-trained surrogate
surr_path = joinpath(PROJECT_ROOT, \"$SURR_CKPT\")
println(\"Loading pre-trained surrogate from \$surr_path ...\")
surr_data   = load(surr_path)
ps          = surr_data[\"ps_cpu\"]
st          = surr_data[\"st_cpu\"]
hidden_dim  = surr_data[\"HIDDEN_DIM\"]
depth       = surr_data[\"DEPTH\"]
param_mins  = surr_data[\"param_mins\"]
param_maxs  = surr_data[\"param_maxs\"]
param_spans = surr_data[\"param_spans\"]

# Load training data
data_path = joinpath(PROJECT_ROOT, \"$DATA_FILE\")
println(\"Loading training data from \$data_path ...\")
data = load(data_path)
all_params  = data[\"all_params\"]
all_signals = data[\"all_signals\"]
bval_list   = data[\"bval_list\"]

param_dim   = size(all_params, 1)
signal_dim  = size(all_signals, 1)
n_total     = size(all_params, 2)

# Normalise parameters
params_norm = (all_params .- param_mins) ./ param_spans

# b0-normalise signals
b0_mask = bval_list .< 0.01
for j in 1:n_total
    b0_mean = mean(all_signals[b0_mask, j])
    all_signals[:, j] ./= max(b0_mean, 1f-6)
end

@printf(\"  Dataset: %d samples, param_dim=%d, signal_dim=%d\\n\",
        n_total, param_dim, signal_dim)

# Rebuild surrogate model (same architecture as pre-trained)
rng = MersenneTwister(42)
surrogate = build_surrogate(
    param_dim  = param_dim,
    signal_dim = signal_dim,
    hidden_dim = hidden_dim,
    depth      = depth,
)

# Move to GPU
if USE_GPU
    ps = ps |> dev
    st = st |> dev
end

# Data function: sample from precomputed dataset
function data_fn(rng, n)
    idx = rand(rng, 1:n_total, n)
    p = params_norm[:, idx]
    s = all_signals[:, idx]
    if USE_GPU
        return dev(p), dev(s)
    end
    return p, s
end

# Define Bloch-Torrey residual with a simple PGSE gradient waveform
# G(t) returns a 3-vector [Gx, Gy, Gz] at time t
# Simple trapezoidal PGSE: gradient along x during [0, 0.3] and [0.5, 0.8]
function pgse_gradient(t)
    g_strength = 0.04f0  # T/m
    if (0.0f0 <= t <= 0.3f0) || (0.5f0 <= t <= 0.8f0)
        return Float32[g_strength, 0f0, 0f0]
    else
        return Float32[0f0, 0f0, 0f0]
    end
end

residual = BlochTorreyResidual(gradient_fn=pgse_gradient)

# Train PINN
println(\"\\nPINN fine-tuning: \$N_STEPS steps, lambda_pde=\$LAMBDA_PDE\")
t0 = time()
ps, st, pinn_losses = train_pinn!(
    surrogate, ps, st, data_fn, residual;
    n_steps       = N_STEPS,
    batch_size    = BATCH_SIZE,
    n_colloc      = N_COLLOC,
    learning_rate = LR,
    lambda_pde    = LAMBDA_PDE,
    print_every   = max(1, N_STEPS ÷ 20),
)
train_time = time() - t0
@printf(\"PINN fine-tuning complete: %.1fs (%.0f steps/s)\\n\", train_time, N_STEPS / train_time)

# Move back to CPU for saving
ps_cpu = USE_GPU ? (ps |> cpu_device()) : ps
st_cpu = USE_GPU ? (st |> cpu_device()) : st

# Evaluate
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

@printf(\"\\nPost-PINN evaluation on %d test samples:\\n\", n_test)
@printf(\"  Median relative error: %.4f (%.2f%%)\\n\", median(rel_errors), median(rel_errors)*100)
@printf(\"  Final L_data: %.6f\\n\", pinn_losses.data[end])
@printf(\"  Final L_pde:  %.6f\\n\", pinn_losses.pde[end])
@printf(\"  Final L_total: %.6f\\n\", pinn_losses.total[end])

# Save checkpoint
ckpt_path = joinpath(PROJECT_ROOT, \"$CHECKPOINT\")
mkpath(dirname(ckpt_path))
@save ckpt_path ps_cpu st_cpu pinn_losses train_time hidden_dim depth param_mins param_maxs param_spans LAMBDA_PDE N_STEPS

println(\"\\nCheckpoint saved: \$ckpt_path\")
"

echo ""
echo "End: $(date)"
