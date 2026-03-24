#!/bin/bash
#SBATCH --job-name=score_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/score_train_%j.out
#SBATCH --error=slurm/logs/score_train_%j.err

# ==========================================================================
# Score network training — conditional denoising score matching.
#
# Uses a trained surrogate as a fast forward model to generate
# (parameters, signals) pairs on the fly during training. The score
# network learns to denoise diffused parameter vectors conditioned on
# observed dMRI signals, enabling posterior sampling at inference time.
#
# Architecture: FiLM-conditioned ScoreNetwork (Perez et al. 2018)
# Training: 100k steps, VP-SDE schedule, epsilon prediction
#
# Usage:
#   sbatch slurm/train_score.sh
#
# Environment variables (optional overrides):
#   SCORE_SURR_CKPT    — path to trained surrogate checkpoint
#                        (default: checkpoints/surrogate.jld2)
#   SCORE_DATA_FILE    — path to merged dataset for bval_list/metadata
#                        (default: data/mcmr_training_data.jld2)
#   SCORE_N_STEPS      — training steps (default: 100000)
#   SCORE_HIDDEN_DIM   — hidden layer width (default: 512)
#   SCORE_DEPTH        — network depth (default: 6)
#   SCORE_COND_DIM     — conditioning embedding dimension (default: 128)
#   SCORE_BATCH_SIZE   — batch size (default: 512)
#   SCORE_LR           — learning rate (default: 3e-4)
#   SCORE_PREDICTION   — prediction target: eps or v (default: eps)
#   SCORE_CHECKPOINT   — output checkpoint path
#                        (default: checkpoints/score_model.jld2)
# ==========================================================================

set -euo pipefail

echo "=========================================="
echo "Score network training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# --- Configuration ---
SURR_CKPT=${SCORE_SURR_CKPT:-checkpoints/surrogate.jld2}
DATA_FILE=${SCORE_DATA_FILE:-data/mcmr_training_data.jld2}
N_STEPS=${SCORE_N_STEPS:-100000}
HIDDEN_DIM=${SCORE_HIDDEN_DIM:-512}
DEPTH=${SCORE_DEPTH:-6}
COND_DIM=${SCORE_COND_DIM:-128}
BATCH_SIZE=${SCORE_BATCH_SIZE:-512}
LR=${SCORE_LR:-3e-4}
PREDICTION=${SCORE_PREDICTION:-eps}
CHECKPOINT=${SCORE_CHECKPOINT:-checkpoints/score_model.jld2}
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}

echo "  Surrogate checkpoint: $SURR_CKPT"
echo "  Data file: $DATA_FILE"
echo "  Architecture: h${HIDDEN_DIM}_d${DEPTH}, cond_dim=${COND_DIM}"
echo "  Steps: $N_STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Prediction: $PREDICTION"
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
include(joinpath(PROJECT_ROOT, \"src/diffusion/score_net.jl\"))
include(joinpath(PROJECT_ROOT, \"src/diffusion/train.jl\"))
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
const COND_DIM   = $COND_DIM
const BATCH_SIZE = $BATCH_SIZE
const LR         = $LR
const PREDICTION = :$PREDICTION

# Load surrogate checkpoint
surr_path = joinpath(PROJECT_ROOT, \"$SURR_CKPT\")
println(\"Loading surrogate from \$surr_path ...\")
surr_data = load(surr_path)
surr_ps = surr_data[\"ps_cpu\"]
surr_st = surr_data[\"st_cpu\"]
surr_hidden = surr_data[\"HIDDEN_DIM\"]
surr_depth  = surr_data[\"DEPTH\"]

# Load dataset metadata (for signal dimensions, bval_list)
data_path = joinpath(PROJECT_ROOT, \"$DATA_FILE\")
println(\"Loading dataset metadata from \$data_path ...\")
data = load(data_path)
bval_list   = data[\"bval_list\"]
signal_dim  = size(data[\"all_signals\"], 1)
param_dim   = size(data[\"all_params\"], 1)
param_mins  = surr_data[\"param_mins\"]
param_maxs  = surr_data[\"param_maxs\"]
param_spans = surr_data[\"param_spans\"]

const B0_MASK = bval_list .< 0.01

@printf(\"  param_dim=%d, signal_dim=%d\\n\", param_dim, signal_dim)

# Reconstruct surrogate model
rng = MersenneTwister(42)
surrogate = build_surrogate(
    param_dim  = param_dim,
    signal_dim = signal_dim,
    hidden_dim = surr_hidden,
    depth      = surr_depth,
)

# Move surrogate to GPU (frozen, no gradient)
if USE_GPU
    surr_ps_dev = surr_ps |> dev
    surr_st_dev = surr_st |> dev
else
    surr_ps_dev = surr_ps
    surr_st_dev = surr_st
end

# Prior: uniform [0, 1] normalised parameters
function sample_prior(rng, n)
    p = rand(rng, Float32, param_dim, n)
    if USE_GPU
        return dev(p)
    end
    return p
end

# Fast surrogate-based forward model with Rician noise + b0 normalisation
function simulator_fn(rng, theta_norm)
    n = size(theta_norm, 2)

    # Surrogate forward pass (no gradient needed)
    signals_clean, _ = surrogate(theta_norm, surr_ps_dev, surr_st_dev)

    # Add Rician noise with variable SNR (SNR in [15, 40])
    snr = rand(rng, Float32, 1, n) .* 25f0 .+ 15f0
    sigma = 1f0 ./ snr
    n1 = randn(rng, Float32, size(signals_clean)) .* sigma
    n2 = randn(rng, Float32, size(signals_clean)) .* sigma

    if USE_GPU
        snr = dev(snr)
        sigma = dev(sigma)
        n1 = dev(n1)
        n2 = dev(n2)
    end

    noisy = @. sqrt((signals_clean + n1)^2 + n2^2)

    # b0-normalise
    b0_mean = mean(noisy[B0_MASK, :], dims=1)
    b0_mean = max.(b0_mean, 1f-6)
    return noisy ./ b0_mean
end

# Build score network
println(\"\\nBuilding score network: h\$(HIDDEN_DIM)_d\$(DEPTH), cond_dim=\$(COND_DIM)\")
score_model = build_score_net(
    param_dim  = param_dim,
    signal_dim = signal_dim,
    hidden_dim = HIDDEN_DIM,
    depth      = DEPTH,
    cond_dim   = COND_DIM,
)
score_ps, score_st = Lux.setup(rng, score_model)

# Move to GPU
if USE_GPU
    score_ps = score_ps |> dev
    score_st = score_st |> dev
end

# Train
schedule = VPSchedule()
println(\"\\nTraining score network: \$N_STEPS steps, prediction=\$PREDICTION\")
t0 = time()
score_ps, score_st, score_losses = train_score!(
    score_model, score_ps, score_st;
    simulator_fn  = simulator_fn,
    prior_fn      = sample_prior,
    schedule      = schedule,
    num_steps     = N_STEPS,
    batch_size    = BATCH_SIZE,
    learning_rate = LR,
    print_every   = max(1, N_STEPS ÷ 20),
    prediction    = PREDICTION,
)
train_time = time() - t0
@printf(\"Training complete: %.1fs (%.0f steps/s)\\n\", train_time, N_STEPS / train_time)

# Save checkpoint (CPU)
score_ps_cpu = USE_GPU ? (score_ps |> cpu_device()) : score_ps
score_st_cpu = USE_GPU ? (score_st |> cpu_device()) : score_st

ckpt_path = joinpath(PROJECT_ROOT, \"$CHECKPOINT\")
mkpath(dirname(ckpt_path))
@save ckpt_path score_ps_cpu score_st_cpu score_losses train_time HIDDEN_DIM DEPTH COND_DIM N_STEPS PREDICTION param_dim signal_dim param_mins param_maxs param_spans

println(\"\\nCheckpoint saved: \$ckpt_path\")
@printf(\"Final score loss: %.6f\\n\", score_losses[end])
"

echo ""
echo "End: $(date)"
