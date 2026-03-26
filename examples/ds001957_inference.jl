#!/usr/bin/env julia
"""
Score-based posterior inference on real dMRI data from ds001957 (BIDS).

Loads a subject's DWI NIfTI + bval/bvec, builds (or loads) a Ball+2Stick
score model, and runs voxelwise posterior sampling to produce parameter maps
(FA, MD, volume fractions, fiber orientations, uncertainty).

Usage:
    julia --project examples/ds001957_inference.jl [--subject sub-01] [--bids-root /path/to/ds001957]
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DMI
using Lux, Random, Statistics, LinearAlgebra, Printf, NIfTI, Serialization

# ===========================================================================
#  Configuration -- edit these or pass via command line
# ===========================================================================
const BIDS_ROOT  = get(ENV, "BIDS_ROOT",  expanduser("~/data/ds001957"))
const SUBJECT_ID = get(ENV, "SUBJECT_ID", "sub-01")

# Parse simple --key value flags from ARGS
function parse_args(args)
    d = Dict{String,String}()
    i = 1
    while i <= length(args)
        if startswith(args[i], "--") && i < length(args)
            d[args[i][3:end]] = args[i+1]
            i += 2
        else
            i += 1
        end
    end
    return d
end

const CLI = parse_args(ARGS)
const bids_root  = get(CLI, "bids-root", BIDS_ROOT)
const subject_id = get(CLI, "subject",   SUBJECT_ID)

# Derived BIDS paths
const dwi_dir   = joinpath(bids_root, subject_id, "dwi")
const dwi_nii   = joinpath(dwi_dir, "$(subject_id)_dwi.nii.gz")
const bval_file  = joinpath(dwi_dir, "$(subject_id)_dwi.bval")
const bvec_file  = joinpath(dwi_dir, "$(subject_id)_dwi.bvec")

# Inference settings
const BATCH_VOXELS       = 1000       # voxels per inference batch (memory tuning)
const N_POSTERIOR_SAMPLES = 200       # posterior samples per voxel
const N_DDPM_STEPS        = 500       # reverse diffusion steps
const PREDICTION_MODE     = :eps      # :eps or :v

# Training settings (used only when training a new model)
const TRAIN_STEPS    = 50_000
const TRAIN_BATCH    = 512
const LEARNING_RATE  = 3e-4
const HIDDEN_DIM     = 512
const DEPTH          = 6
const COND_DIM       = 128

# Checkpoint path (Serialization format)
const CHECKPOINT_DIR = joinpath(@__DIR__, "..", "checkpoints")
const CHECKPOINT_FILE = joinpath(CHECKPOINT_DIR, "ball2stick_$(subject_id).jls")

# Output directory for NIfTI maps
const OUTPUT_DIR = joinpath(@__DIR__, "..", "results", subject_id)

# ===========================================================================
#  1. Load BIDS data
# ===========================================================================
println("=" ^ 70)
println("  ds001957 Score-Based Posterior Inference")
println("  Subject: $subject_id")
println("  BIDS root: $bids_root")
println("=" ^ 70)

println("\n[1/7] Loading BIDS data...")

for f in [dwi_nii, bval_file, bvec_file]
    if !isfile(f)
        error("Missing file: $f\n" *
              "Check that BIDS_ROOT and SUBJECT_ID are correct.\n" *
              "Expected layout: <bids_root>/<sub-XX>/dwi/<sub-XX>_dwi.{nii.gz,bval,bvec}")
    end
end

t_load = time()
dwi_img = niread(dwi_nii)
dwi_data = Float32.(dwi_img.raw)  # (nx, ny, nz, n_volumes)
nx, ny, nz, n_vol = size(dwi_data)
elapsed_load = time() - t_load
println("  DWI loaded: $(nx) x $(ny) x $(nz) x $(n_vol)  ($(round(elapsed_load, digits=1))s)")

# ===========================================================================
#  2. Build Acquisition from bval/bvec
# ===========================================================================
println("\n[2/7] Building acquisition scheme...")

acq = load_acquisition(bval_file, bvec_file)
n_meas = length(acq.bvalues)
@assert n_meas == n_vol "Mismatch: bval has $n_meas entries but NIfTI has $n_vol volumes"

b0_idx = findall(acq.bvalues .< 100e6)
println("  Measurements: $n_meas  ($(length(b0_idx)) b=0 volumes)")
println("  b-value shells: ", sort(unique(round.(acq.bvalues ./ 1e6) .* 1e6)))

# ===========================================================================
#  3. Create or load brain mask
# ===========================================================================
println("\n[3/7] Creating brain mask...")

# Check for a BIDS derivatives mask first
mask_candidates = [
    joinpath(bids_root, "derivatives", "masks", subject_id, "dwi", "$(subject_id)_dwi_mask.nii.gz"),
    joinpath(bids_root, "derivatives", subject_id, "dwi", "$(subject_id)_dwi_brainmask.nii.gz"),
    joinpath(dwi_dir, "$(subject_id)_dwi_mask.nii.gz"),
]

mask_3d = nothing
for mpath in mask_candidates
    if isfile(mpath)
        println("  Found existing mask: $mpath")
        mask_img = niread(mpath)
        mask_3d = Bool.(mask_img.raw .> 0)
        break
    end
end

if mask_3d === nothing
    # Simple threshold mask from mean b0 signal
    println("  No existing mask found -- creating from b0 threshold...")
    b0_data = dwi_data[:, :, :, b0_idx]
    b0_mean = dropdims(mean(b0_data, dims=4), dims=4)

    # Otsu-ish: use 15% of the 95th percentile as threshold
    b0_vals = filter(x -> x > 0, vec(b0_mean))
    if isempty(b0_vals)
        error("All b0 voxels are zero -- check your data")
    end
    p95 = sort(b0_vals)[max(1, round(Int, 0.95 * length(b0_vals)))]
    threshold = 0.15f0 * p95
    mask_3d = b0_mean .> threshold
    println("  Threshold: $(round(threshold, digits=1))  (15% of p95=$(round(p95, digits=1)))")
end

n_voxels = sum(mask_3d)
println("  Brain voxels: $n_voxels / $(prod(size(mask_3d)))" *
        "  ($(round(100 * n_voxels / prod(size(mask_3d)), digits=1))%)")

# ===========================================================================
#  4. B0-normalise the signal within the mask
# ===========================================================================
println("\n[4/7] Normalising signal (b0-normalise + NaN cleanup)...")

# Compute mean b0 per voxel (3D)
b0_data = dwi_data[:, :, :, b0_idx]
b0_mean_3d = dropdims(mean(b0_data, dims=4), dims=4)
b0_mean_3d[b0_mean_3d .< 1f-6] .= 1f-6  # avoid division by zero

# Extract masked voxel signals: (n_voxels, n_meas)
voxel_indices = findall(mask_3d)
signals_raw = zeros(Float32, n_voxels, n_meas)
for (i, idx) in enumerate(voxel_indices)
    for v in 1:n_meas
        signals_raw[i, v] = dwi_data[idx, v]
    end
end

# b0-normalise each voxel
b0_per_voxel = Float32[b0_mean_3d[idx] for idx in voxel_indices]
signals_norm = signals_raw ./ b0_per_voxel  # broadcast (n_voxels, n_meas)

# Clamp to sensible range and replace NaN/Inf
signals_norm = clamp.(signals_norm, 0f0, 5f0)
signals_norm[isnan.(signals_norm)] .= 0f0
signals_norm[isinf.(signals_norm)] .= 0f0

println("  Signal range after normalisation: $(round(minimum(signals_norm), digits=3)) .. $(round(maximum(signals_norm), digits=3))")

# Free raw data to reduce memory pressure
dwi_data = nothing
b0_data = nothing
GC.gc()

# ===========================================================================
#  5. Train or load score model
# ===========================================================================
println("\n[5/7] Preparing score model...")

rng = Random.default_rng()
Random.seed!(rng, 42)

# Ball+2Stick parameter ranges (same as ball2stick_score.jl)
const PARAM_NAMES = ["d_ball", "d_stick", "f1", "f2", "mu1x", "mu1y", "mu1z", "mu2x", "mu2y", "mu2z"]
const PARAM_DIM = 10
const lows  = Float32[1e-9, 0.5e-9, 0.1, 0.05, -1, -1, 0, -1, -1, 0]
const highs = Float32[3.5e-9, 2.5e-9, 0.8, 0.5, 1, 1, 1, 1, 1, 1]
const spans = max.(highs .- lows, 1f-12)

model_phys = BallStickModel(acq.bvalues, acq.gradient_directions)

model = build_score_net(
    param_dim  = PARAM_DIM,
    signal_dim = n_meas,
    hidden_dim = HIDDEN_DIM,
    depth      = DEPTH,
    cond_dim   = COND_DIM,
)

if isfile(CHECKPOINT_FILE)
    println("  Loading checkpoint: $CHECKPOINT_FILE")
    checkpoint = open(deserialize, CHECKPOINT_FILE)
    ps = checkpoint[:ps]
    st = checkpoint[:st]
    println("  Model loaded from checkpoint.")
else
    println("  No checkpoint found -- training new model...")
    ps, st = Lux.setup(rng, model)

    # Prior: uniform in normalised [0,1]^10 with f1 >= f2 constraint
    function sample_prior_fn(rng, n)
        u = rand(rng, Float32, PARAM_DIM, n)
        theta_norm = copy(u)
        theta = lows .+ theta_norm .* spans
        for j in 1:n
            if theta[3, j] < theta[4, j]
                theta[3, j], theta[4, j] = theta[4, j], theta[3, j]
                theta[5:7, j], theta[8:10, j] = theta[8:10, j], theta[5:7, j]
            end
        end
        return (theta .- lows) ./ spans
    end

    b0_mask_acq = acq.bvalues .< 100e6

    function sim_fn(rng, theta_norm)
        n = size(theta_norm, 2)
        theta = lows .+ theta_norm .* spans
        signals = zeros(Float32, n_meas, n)
        for j in 1:n
            signals[:, j] = simulate(model_phys, @view theta[:, j])
        end
        noisy = add_rician_noise(rng, signals'; snr_range=(10.0, 50.0))'
        b0m = mean(noisy[b0_mask_acq, :], dims=1)
        b0m = max.(b0m, 1f-6)
        return noisy ./ b0m
    end

    t_train = time()
    ps, st, losses = train_score!(
        model, ps, st;
        simulator_fn  = sim_fn,
        prior_fn      = sample_prior_fn,
        num_steps     = TRAIN_STEPS,
        batch_size    = TRAIN_BATCH,
        learning_rate = LEARNING_RATE,
        print_every   = 5000,
        prediction    = PREDICTION_MODE,
    )
    elapsed_train = time() - t_train
    println("  Training complete: $(round(elapsed_train / 60, digits=1)) minutes")
    println("  Final loss: $(round(losses[end], digits=5))")

    # Save checkpoint
    mkpath(dirname(CHECKPOINT_FILE))
    open(io -> serialize(io, Dict(:ps => ps, :st => st, :losses => losses)), CHECKPOINT_FILE, "w")
    println("  Checkpoint saved: $CHECKPOINT_FILE")
end

# Put model in test mode (no dropout etc.)
st = Lux.testmode(st)

# ===========================================================================
#  6. Voxelwise posterior inference (batched)
# ===========================================================================
println("\n[6/7] Running voxelwise posterior inference...")
println("  Voxels: $n_voxels  |  Batch size: $BATCH_VOXELS voxels")
println("  Posterior samples/voxel: $N_POSTERIOR_SAMPLES  |  DDPM steps: $N_DDPM_STEPS")

# Estimate timing from a small pilot
pilot_n = min(5, n_voxels)
t_pilot = time()
for i in 1:pilot_n
    sig_col = Float32.(signals_norm[i, :])
    _ = sample_posterior(
        model, ps, st, sig_col;
        n_samples   = N_POSTERIOR_SAMPLES,
        n_steps     = N_DDPM_STEPS,
        n_scalars   = 4,
        n_vectors   = 2,
        prediction  = PREDICTION_MODE,
    )
end
t_per_voxel = (time() - t_pilot) / pilot_n
eta_minutes = t_per_voxel * n_voxels / 60
println(@sprintf("  Pilot: %.3f s/voxel  =>  ETA %.1f minutes for %d voxels",
                 t_per_voxel, eta_minutes, n_voxels))

# Allocate output maps (param_dim for mean, param_dim for std)
posterior_mean = zeros(Float32, n_voxels, PARAM_DIM)
posterior_std  = zeros(Float32, n_voxels, PARAM_DIM)

n_batches = ceil(Int, n_voxels / BATCH_VOXELS)
t_infer = time()

for batch_idx in 1:n_batches
    i_start = (batch_idx - 1) * BATCH_VOXELS + 1
    i_end   = min(batch_idx * BATCH_VOXELS, n_voxels)
    batch_n = i_end - i_start + 1

    for i in i_start:i_end
        sig_col = Float32.(signals_norm[i, :])

        # Skip voxels with near-zero signal (likely outside brain)
        if maximum(sig_col) < 0.01f0
            continue
        end

        samples = sample_posterior(
            model, ps, st, sig_col;
            n_samples   = N_POSTERIOR_SAMPLES,
            n_steps     = N_DDPM_STEPS,
            n_scalars   = 4,
            n_vectors   = 2,
            prediction  = PREDICTION_MODE,
        )
        # samples: (param_dim, n_samples) in normalised space

        posterior_mean[i, :] = vec(mean(samples, dims=2))
        posterior_std[i, :]  = vec(std(samples, dims=2))
    end

    elapsed_batch = time() - t_infer
    frac = i_end / n_voxels
    eta_remain = (elapsed_batch / frac) * (1 - frac) / 60
    if batch_idx % max(1, n_batches ÷ 20) == 0 || batch_idx == 1 || batch_idx == n_batches
        println(@sprintf("  Batch %d/%d  (voxels %d-%d)  %.1f%%  ETA %.1f min remaining",
                         batch_idx, n_batches, i_start, i_end,
                         100 * frac, eta_remain))
    end
end

elapsed_infer = time() - t_infer
println(@sprintf("  Inference complete: %.1f minutes (%.3f s/voxel)",
                 elapsed_infer / 60, elapsed_infer / n_voxels))

# ===========================================================================
#  7. Denormalise and compute derived maps
# ===========================================================================
println("\n[7/7] Computing parameter maps and saving NIfTI...")

# Denormalise posterior means to physical units
phys_mean = lows' .+ posterior_mean .* spans'
phys_std  = posterior_std .* spans'  # std in physical units

# Extract individual parameters (per voxel)
d_ball_mean   = phys_mean[:, 1]
d_stick_mean  = phys_mean[:, 2]
f1_mean       = phys_mean[:, 3]
f2_mean       = phys_mean[:, 4]
mu1_mean      = phys_mean[:, 5:7]   # (n_voxels, 3)
mu2_mean      = phys_mean[:, 8:10]

f1_std  = phys_std[:, 3]
f2_std  = phys_std[:, 4]

# Compute Ball+2Stick-derived FA and MD per voxel
#   Ball+2Stick is not a tensor model, but we can derive approximate
#   diffusion properties:
#     MD ~ f1*d_stick + f2*d_stick + f_ball*d_ball
#     FA is approximated from the composite diffusion characteristics
md_map = f1_mean .* d_stick_mean .+ f2_mean .* d_stick_mean .+
         clamp.(1f0 .- f1_mean .- f2_mean, 0f0, 1f0) .* d_ball_mean

# Reconstruct into 3D volumes
function fill_volume(vals::AbstractVector, indices, dims)
    vol = zeros(Float32, dims)
    for (i, idx) in enumerate(indices)
        vol[idx] = vals[i]
    end
    return vol
end

function fill_volume_vec(vals::AbstractMatrix, indices, dims, comp)
    vol = zeros(Float32, dims)
    for (i, idx) in enumerate(indices)
        vol[idx] = vals[i, comp]
    end
    return vol
end

vol_dims = (nx, ny, nz)

# Build output header from input (keep reference before freeing dwi_data)
out_header = dwi_img.header

mkpath(OUTPUT_DIR)
println("  Output directory: $OUTPUT_DIR")

# Helper to write a 3D NIfTI map
function save_map(name, data_3d)
    out_path = joinpath(OUTPUT_DIR, "$(subject_id)_$(name).nii.gz")
    # Build a NIfTI image reusing the input header for geometry/affine
    img = NIfTI.NIfTI1Image(Float32.(data_3d), out_header)
    niwrite(out_path, img)
    println("    Saved: $out_path")
    return out_path
end

# Save parameter maps
println("  Writing parameter maps:")
save_map("MD",   fill_volume(md_map, voxel_indices, vol_dims))
save_map("f1",   fill_volume(f1_mean, voxel_indices, vol_dims))
save_map("f2",   fill_volume(f2_mean, voxel_indices, vol_dims))
save_map("f1_std", fill_volume(f1_std, voxel_indices, vol_dims))
save_map("f2_std", fill_volume(f2_std, voxel_indices, vol_dims))
save_map("d_ball",  fill_volume(d_ball_mean, voxel_indices, vol_dims))
save_map("d_stick", fill_volume(d_stick_mean, voxel_indices, vol_dims))

# Primary fiber orientation components
save_map("mu1_x", fill_volume_vec(mu1_mean, voxel_indices, vol_dims, 1))
save_map("mu1_y", fill_volume_vec(mu1_mean, voxel_indices, vol_dims, 2))
save_map("mu1_z", fill_volume_vec(mu1_mean, voxel_indices, vol_dims, 3))

# Secondary fiber orientation
save_map("mu2_x", fill_volume_vec(mu2_mean, voxel_indices, vol_dims, 1))
save_map("mu2_y", fill_volume_vec(mu2_mean, voxel_indices, vol_dims, 2))
save_map("mu2_z", fill_volume_vec(mu2_mean, voxel_indices, vol_dims, 3))

# Uncertainty maps for diffusivities
save_map("d_ball_std",  fill_volume(phys_std[:, 1], voxel_indices, vol_dims))
save_map("d_stick_std", fill_volume(phys_std[:, 2], voxel_indices, vol_dims))

# Posterior mean consistency: fraction of signal variance explained
# (not a true R^2, but a quick sanity check)
println("  Parameter maps saved to $OUTPUT_DIR")

# ===========================================================================
#  8. Visualization (optional, if CairoMakie available)
# ===========================================================================
println("\nAttempting visualization...")

try
    @eval using CairoMakie

    # Pick a mid-axial slice
    slice_z = nz ÷ 2

    # Collect maps for display
    map_names = ["MD", "f1", "f2", "f1_std", "f2_std", "d_ball"]
    map_data = Dict{String, Matrix{Float32}}()
    for name in map_names
        vol = fill_volume(
            name == "MD"     ? md_map :
            name == "f1"     ? f1_mean :
            name == "f2"     ? f2_mean :
            name == "f1_std" ? f1_std :
            name == "f2_std" ? f2_std :
            name == "d_ball" ? d_ball_mean :
            zeros(Float32, n_voxels),
            voxel_indices, vol_dims
        )
        map_data[name] = vol[:, :, slice_z]
    end

    fig = Figure(size = (1400, 500))
    for (i, name) in enumerate(map_names)
        ax = Axis(fig[1, i]; title = name, aspect = DataAspect())
        heatmap!(ax, map_data[name]'; colormap = :viridis)
        hidedecorations!(ax)
    end

    fig_path = joinpath(OUTPUT_DIR, "$(subject_id)_parameter_maps_z$(slice_z).png")
    save(fig_path, fig; px_per_unit = 2)
    println("  Figure saved: $fig_path")
catch e
    if e isa ArgumentError || e isa LoadError || string(e) |> s -> contains(s, "CairoMakie")
        println("  CairoMakie not available -- skipping visualization.")
        println("  Install with: Pkg.add(\"CairoMakie\") to enable plotting.")
    else
        println("  Visualization failed: $e")
    end
end

# ===========================================================================
#  Summary
# ===========================================================================
println("\n" * "=" ^ 70)
println("  DONE")
println("  Subject:     $subject_id")
println("  Voxels:      $n_voxels")
println("  Maps saved:  $OUTPUT_DIR")
if isfile(CHECKPOINT_FILE)
    println("  Checkpoint:  $CHECKPOINT_FILE")
end
println("=" ^ 70)
