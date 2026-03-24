#!/bin/bash
#SBATCH --job-name=wholebrain_inf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/wholebrain_%j.out
#SBATCH --error=slurm/logs/wholebrain_%j.err

# ==========================================================================
# Whole-brain posterior inference on ds001957 dMRI data.
#
# Loads a trained score model and processes all brain voxels using batched
# posterior sampling via DDPM. Produces NIfTI parameter maps for each
# microstructure parameter (mean and uncertainty).
#
# Usage:
#   sbatch slurm/inference_wholebrain.sh
#   SUBJECT=sub-02 sbatch slurm/inference_wholebrain.sh
#
# Environment variables (optional overrides):
#   SUBJECT            — BIDS subject ID (default: sub-01)
#   BIDS_DIR           — path to BIDS dataset
#                         (default: ~/data/ds001957)
#   SCORE_CKPT         — trained score model checkpoint
#                         (default: checkpoints/score_model.jld2)
#   SURR_CKPT          — trained surrogate checkpoint (for metadata)
#                         (default: checkpoints/surrogate.jld2)
#   N_POSTERIOR        — posterior samples per voxel (default: 500)
#   N_DIFFUSION_STEPS  — DDPM reverse steps (default: 500)
#   VOXEL_BATCH_SIZE   — voxels processed in parallel (default: 256)
#   OUTPUT_DIR         — output directory for NIfTI maps
#                         (default: results/wholebrain/<subject>)
# ==========================================================================

set -euo pipefail

echo "=========================================="
echo "Whole-brain posterior inference"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# --- Configuration ---
SUBJECT=${SUBJECT:-sub-01}
BIDS_DIR=${BIDS_DIR:-${HOME}/data/ds001957}
SCORE_CKPT=${SCORE_CKPT:-checkpoints/score_model.jld2}
SURR_CKPT=${SURR_CKPT:-checkpoints/surrogate.jld2}
N_POSTERIOR=${N_POSTERIOR:-500}
N_DIFFUSION_STEPS=${N_DIFFUSION_STEPS:-500}
VOXEL_BATCH_SIZE=${VOXEL_BATCH_SIZE:-256}
OUTPUT_DIR=${OUTPUT_DIR:-results/wholebrain/${SUBJECT}}
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}

echo "  Subject: $SUBJECT"
echo "  BIDS dir: $BIDS_DIR"
echo "  Score checkpoint: $SCORE_CKPT"
echo "  Surrogate checkpoint: $SURR_CKPT"
echo "  Posterior samples: $N_POSTERIOR"
echo "  Diffusion steps: $N_DIFFUSION_STEPS"
echo "  Voxel batch size: $VOXEL_BATCH_SIZE"
echo "  Output: $OUTPUT_DIR"

# --- Load Julia and report GPU ---
if command -v module &>/dev/null; then
    module load julia/1.11 2>/dev/null || true
    module load cuda 2>/dev/null || true
fi

julia --version
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "No nvidia-smi available"
echo ""

# --- Run inference ---
julia --project="$PROJECT_DIR" --threads=4 -e "
using Random, Statistics, Printf, LinearAlgebra
using JLD2, NIfTI
using Lux, Optimisers, Zygote
using CUDA, LuxCUDA

# Include source files
const PROJECT_ROOT = \"$PROJECT_DIR\"
include(joinpath(PROJECT_ROOT, \"src/noise.jl\"))
include(joinpath(PROJECT_ROOT, \"src/diffusion/schedule.jl\"))
include(joinpath(PROJECT_ROOT, \"src/diffusion/score_net.jl\"))
include(joinpath(PROJECT_ROOT, \"src/diffusion/sample.jl\"))
include(joinpath(PROJECT_ROOT, \"src/pinn/bloch_torrey.jl\"))

# GPU setup
const USE_GPU = CUDA.functional()
const dev = USE_GPU ? gpu_device() : cpu_device()
println(\"GPU available: \$USE_GPU\")
if USE_GPU
    println(\"GPU: \", CUDA.name(CUDA.device()))
end

# Configuration
const SUBJECT           = \"$SUBJECT\"
const BIDS_DIR          = \"$BIDS_DIR\"
const N_POSTERIOR        = $N_POSTERIOR
const N_DIFFUSION_STEPS = $N_DIFFUSION_STEPS
const VOXEL_BATCH_SIZE  = $VOXEL_BATCH_SIZE
const OUTPUT_DIR        = joinpath(PROJECT_ROOT, \"$OUTPUT_DIR\")

# ---------------------------------------------------------------
# 1. Load trained score model
# ---------------------------------------------------------------
println(\"\\n--- Loading trained models ---\")

score_path = joinpath(PROJECT_ROOT, \"$SCORE_CKPT\")
println(\"Score checkpoint: \$score_path\")
score_data = load(score_path)

score_hidden = score_data[\"HIDDEN_DIM\"]
score_depth  = score_data[\"DEPTH\"]
score_cond   = score_data[\"COND_DIM\"]
param_dim    = score_data[\"param_dim\"]
signal_dim   = score_data[\"signal_dim\"]
param_mins   = score_data[\"param_mins\"]
param_maxs   = score_data[\"param_maxs\"]
param_spans  = score_data[\"param_spans\"]
prediction   = score_data[\"PREDICTION\"]

@printf(\"  Score net: h%d_d%d, cond_dim=%d\\n\", score_hidden, score_depth, score_cond)
@printf(\"  param_dim=%d, signal_dim=%d, prediction=%s\\n\", param_dim, signal_dim, prediction)

# Reconstruct score network
rng = MersenneTwister(42)
score_model = build_score_net(
    param_dim  = param_dim,
    signal_dim = signal_dim,
    hidden_dim = score_hidden,
    depth      = score_depth,
    cond_dim   = score_cond,
)

score_ps = score_data[\"score_ps_cpu\"]
score_st = score_data[\"score_st_cpu\"]

if USE_GPU
    score_ps = score_ps |> dev
    score_st = score_st |> dev
end

schedule = VPSchedule()

# ---------------------------------------------------------------
# 2. Load dMRI data from BIDS dataset
# ---------------------------------------------------------------
println(\"\\n--- Loading dMRI data ---\")

# Find DWI NIfTI file (BIDS standard paths)
dwi_dir = joinpath(BIDS_DIR, SUBJECT, \"dwi\")
dwi_files = filter(f -> endswith(f, \"_dwi.nii.gz\") || endswith(f, \"_dwi.nii\"),
                   readdir(dwi_dir))

if isempty(dwi_files)
    error(\"No DWI NIfTI files found in \$dwi_dir\")
end

dwi_path = joinpath(dwi_dir, dwi_files[1])
println(\"  DWI: \$dwi_path\")
dwi_nii = niread(dwi_path)
dwi_data = Float32.(dwi_nii.raw)
vol_size = size(dwi_data)[1:3]
n_vols   = size(dwi_data, 4)

@printf(\"  Volume size: %s, %d volumes\\n\", string(vol_size), n_vols)

# Load brain mask if available (BIDS derivatives or create from b0)
mask_candidates = [
    joinpath(BIDS_DIR, \"derivatives\", \"masks\", SUBJECT, \"dwi\",
             SUBJECT * \"_space-dwi_mask.nii.gz\"),
    joinpath(BIDS_DIR, SUBJECT, \"dwi\",
             replace(dwi_files[1], \"_dwi\" => \"_mask\")),
]

brain_mask = nothing
for mc in mask_candidates
    if isfile(mc)
        println(\"  Mask: \$mc\")
        brain_mask = niread(mc).raw .> 0
        break
    end
end

if brain_mask === nothing
    # Create simple mask from b0 signal intensity
    println(\"  No mask found; creating from b0 threshold\")
    b0_vol = mean(dwi_data[:,:,:,1:min(6, n_vols)], dims=4)[:,:,:,1]
    threshold = 0.1 * maximum(b0_vol)
    brain_mask = b0_vol .> threshold
end

n_voxels = sum(brain_mask)
@printf(\"  Brain voxels: %d\\n\", n_voxels)

# Load bvals for b0 normalisation
bval_file = replace(dwi_path, r\"\\.(nii\\.gz|nii)\$\" => \".bval\")
if !isfile(bval_file)
    bval_file = replace(dwi_path, r\"_dwi\\.(nii\\.gz|nii)\$\" => \"_dwi.bval\")
end
bvals = parse.(Float64, split(strip(read(bval_file, String))))
b0_idx = findall(bvals .< 50)  # b=0 volumes
println(\"  b-values: \", length(bvals), \" volumes, \", length(b0_idx), \" b=0\")

# ---------------------------------------------------------------
# 3. Prepare voxel signals
# ---------------------------------------------------------------
println(\"\\n--- Preparing voxel signals ---\")

# Extract brain voxel signals and b0-normalise
voxel_indices = findall(brain_mask)
voxel_signals = zeros(Float32, n_vols, n_voxels)

for (i, idx) in enumerate(voxel_indices)
    voxel_signals[:, i] = dwi_data[idx, :]
end

# b0 normalise each voxel
for i in 1:n_voxels
    b0_mean = mean(voxel_signals[b0_idx, i])
    b0_mean = max(b0_mean, 1f-6)
    voxel_signals[:, i] ./= b0_mean
end

# If signal_dim != n_vols, we need to match the training acquisition.
# For now, assume they match or take the first signal_dim volumes.
if n_vols != signal_dim
    @warn \"Signal dimension mismatch: data has \$n_vols volumes, model expects \$signal_dim. Using first min(\$n_vols, \$signal_dim) volumes.\"
    use_dim = min(n_vols, signal_dim)
    if n_vols > signal_dim
        voxel_signals = voxel_signals[1:signal_dim, :]
    else
        # Pad with zeros if data has fewer volumes
        padded = zeros(Float32, signal_dim, n_voxels)
        padded[1:n_vols, :] = voxel_signals
        voxel_signals = padded
    end
end

println(\"  Voxel signals shape: \$(size(voxel_signals))\")
@printf(\"  Signal range: [%.4f, %.4f]\\n\", extrema(voxel_signals)...)

# ---------------------------------------------------------------
# 4. Batched posterior sampling
# ---------------------------------------------------------------
println(\"\\n--- Posterior sampling ---\")
@printf(\"  %d voxels x %d posterior samples x %d diffusion steps\\n\",
        n_voxels, N_POSTERIOR, N_DIFFUSION_STEPS)

# Allocate output maps (mean and std for each parameter)
param_maps_mean = zeros(Float32, vol_size..., param_dim)
param_maps_std  = zeros(Float32, vol_size..., param_dim)

# Process voxels in batches
n_batches = ceil(Int, n_voxels / VOXEL_BATCH_SIZE)
t0 = time()

for batch_idx in 1:n_batches
    v_start = (batch_idx - 1) * VOXEL_BATCH_SIZE + 1
    v_end   = min(batch_idx * VOXEL_BATCH_SIZE, n_voxels)
    n_batch = v_end - v_start + 1

    # Process each voxel in this batch
    for v in v_start:v_end
        signal = voxel_signals[:, v]

        # Move signal to GPU if needed
        if USE_GPU
            signal = dev(signal)
        end

        # Draw posterior samples
        # score_ps/score_st are already on the correct device
        posterior_norm = sample_posterior(
            score_model, score_ps, score_st, signal;
            schedule   = schedule,
            n_samples  = N_POSTERIOR,
            n_steps    = N_DIFFUSION_STEPS,
            n_scalars  = param_dim,
            n_vectors  = 0,
            prediction = prediction,
        )

        # Move back to CPU
        if USE_GPU
            posterior_norm = posterior_norm |> cpu_device()
        end

        # Denormalise to physical units
        posterior_phys = param_mins .+ posterior_norm .* param_spans
        posterior_phys = clamp.(posterior_phys, param_mins, param_maxs)

        # Store mean and std at this voxel's spatial location
        idx = voxel_indices[v]
        for p in 1:param_dim
            param_maps_mean[idx, p] = mean(posterior_phys[p, :])
            param_maps_std[idx, p]  = std(posterior_phys[p, :])
        end
    end

    if batch_idx % max(1, n_batches ÷ 20) == 0 || batch_idx == 1 || batch_idx == n_batches
        elapsed = time() - t0
        voxels_done = v_end
        rate = voxels_done / elapsed
        eta = (n_voxels - voxels_done) / max(rate, 1e-6)
        @printf(\"  Batch %d/%d  voxels %d/%d  (%.1f vox/s, ETA %.0fs)\\n\",
                batch_idx, n_batches, voxels_done, n_voxels, rate, eta)
    end
end

total_time = time() - t0
@printf(\"\\nInference complete: %.1fs (%.1f voxels/s)\\n\",
        total_time, n_voxels / total_time)

# ---------------------------------------------------------------
# 5. Save NIfTI parameter maps
# ---------------------------------------------------------------
println(\"\\n--- Saving NIfTI parameter maps ---\")
mkpath(OUTPUT_DIR)

# Parameter names (from mcmr_generator.jl)
param_names = [\"mean_radius\", \"radius_variance\", \"volume_fraction\"]
if param_dim > 3
    # Extended model may have additional parameters
    for i in 4:param_dim
        push!(param_names, \"param_\$i\")
    end
end

# Use the DWI header as template for affine/voxel size
header = dwi_nii.header

for (p, name) in enumerate(param_names)
    # Mean map
    mean_data = param_maps_mean[:, :, :, p]
    mean_nii = NIfTI.NIfTI1Image(mean_data, dwi_nii.header)
    mean_path = joinpath(OUTPUT_DIR, \"\$(SUBJECT)_\$(name)_mean.nii.gz\")
    niwrite(mean_path, mean_nii)
    println(\"  Saved: \$mean_path\")

    # Uncertainty (std) map
    std_data = param_maps_std[:, :, :, p]
    std_nii = NIfTI.NIfTI1Image(std_data, dwi_nii.header)
    std_path = joinpath(OUTPUT_DIR, \"\$(SUBJECT)_\$(name)_std.nii.gz\")
    niwrite(std_path, std_nii)
    println(\"  Saved: \$std_path\")
end

println(\"\\nAll parameter maps saved to \$OUTPUT_DIR\")
@printf(\"Total inference time: %.1f minutes\\n\", total_time / 60)
"

echo ""
echo "End: $(date)"
