#!/usr/bin/env julia
# WAND sub-08033 ses-02: Quantitative Magnetization Transfer (qMT) processing
#
# Uses MRIgeneralizedBloch.jl to fit the generalized Bloch model to
# multi-flip-angle, multi-offset qMT data from CUBRIC's tfl_qMT_v09 sequence.
#
# Acquisition (from BIDS JSON sidecars):
#   - Sequence: tfl_qMT_v09 (3D turbo FLASH with MT preparation)
#   - TR = 55 ms, TE = 2.1 ms
#   - 3 MT pulse flip angles: 332°, 628°, 333° (cumulative)
#   - MT offsets: 56360, 47180, 12060, 2750, 2770, 2790, 2890, 1000 Hz
#   - MT-off reference (FA = 5°, no MT)
#   - Resolution: 1.72 mm iso, 104×128×128
#   - Field: 3T Siemens Connectom-A
#
# NOTE: The MT pulse duration (TRF) is not in the BIDS sidecars.
#       For CUBRIC's tfl_qMT_v09, the standard pulse duration is ~12 ms
#       (Fermi pulse). This needs verification from the CUBRIC protocol.
#       The value below is an educated estimate.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))  # activate dmijl project root

using NIfTI
using MRIgeneralizedBloch
using MriResearchTools
using Statistics
using Printf
using LinearAlgebra

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const SUB = "sub-08033"
const SES = "ses-02"
const RAWDIR = "/data/raw/wand/$SUB/$SES/anat"
const OUTDIR = "/data/datasets/wand/derivatives/qmt/$SUB/$SES"

mkpath(OUTDIR)

# QMT acquisition parameters from BIDS JSON sidecars
const TR = 0.055  # seconds (from RepetitionTime)

# MT pulse flip angles (cumulative, in degrees) and offset frequencies (Hz)
# Ordered to match file loading order below
const QMT_PROTOCOL = [
    # (label,          FA_deg, MT_offset_Hz)
    ("flip-1_mt-1",    332,    56360),
    ("flip-1_mt-2",    332,    1000),
    ("flip-2_mt-1",    628,    47180),
    ("flip-2_mt-2",    628,    12060),
    ("flip-2_mt-3",    628,    2750),
    ("flip-2_mt-4",    628,    2770),
    ("flip-2_mt-5",    628,    2790),
    ("flip-2_mt-6",    628,    2890),
    ("flip-2_mt-7",    628,    1000),  # average of run-1 and run-2
    ("flip-3_mt-1",    333,    1000),
    ("mt-off",         5,      0),     # reference (no MT)
]

# MT pulse duration — ESTIMATED, needs CUBRIC protocol confirmation
# tfl_qMT_v09 typically uses a 12 ms Fermi pulse
const TRF_ESTIMATED = 12e-3  # seconds

println("=" ^ 60)
println("WAND qMT Processing: $SUB $SES")
println("=" ^ 60)

# ---------------------------------------------------------------------------
# 1. Load qMT volumes
# ---------------------------------------------------------------------------
println("\n[1/4] Loading qMT volumes...")

function load_qmt_volume(label)
    if label == "mt-off"
        path = joinpath(RAWDIR, "$(SUB)_$(SES)_$(label)_part-mag_QMT.nii.gz")
    else
        path = joinpath(RAWDIR, "$(SUB)_$(SES)_$(label)_part-mag_QMT.nii.gz")
    end
    if !isfile(path)
        # Try with run-1 for repeated acquisitions
        path_r1 = replace(path, "_part-mag_QMT" => "_part-mag_run-1_QMT")
        path_r2 = replace(path, "_part-mag_QMT" => "_part-mag_run-2_QMT")
        if isfile(path_r1) && isfile(path_r2)
            v1 = Float64.(niread(path_r1))
            v2 = Float64.(niread(path_r2))
            return (v1 .+ v2) ./ 2  # average repeated measurements
        elseif isfile(path_r1)
            return Float64.(niread(path_r1))
        end
    end
    return Float64.(niread(path))
end

volumes = Dict{String, Array{Float64,3}}()
for (label, fa, offset) in QMT_PROTOCOL
    vol = load_qmt_volume(label)
    volumes[label] = vol
    @printf("  %-20s FA=%4d°  offset=%6d Hz  size=%s\n", label, fa, offset, string(size(vol)))
end

nx, ny, nz = size(first(values(volumes)))
println("  Loaded $(length(volumes)) volumes, $(nx)×$(ny)×$(nz)")

# ---------------------------------------------------------------------------
# 2. Brain mask from MT-off reference
# ---------------------------------------------------------------------------
println("\n[2/4] Computing brain mask from MT-off reference...")

mt_off = volumes["mt-off"]
mask = robustmask(Float32.(mt_off))
mask_brain = brain_mask(mask)
n_brain = sum(mask_brain)
@printf("  Brain voxels: %d / %d (%.1f%%)\n", n_brain, length(mask_brain),
        100 * n_brain / length(mask_brain))

hdr = header(niread(joinpath(RAWDIR, "$(SUB)_$(SES)_mt-off_part-mag_QMT.nii.gz")))
savenii(mask_brain, joinpath(OUTDIR, "brain_mask.nii.gz"); header=hdr)

# ---------------------------------------------------------------------------
# 3. Prepare data for fitting
# ---------------------------------------------------------------------------
println("\n[3/4] Preparing qMT data for generalized Bloch fitting...")

# Stack signals in protocol order (excluding mt-off reference)
mt_labels = [l for (l, _, _) in QMT_PROTOCOL if l != "mt-off"]
n_mt = length(mt_labels)

# Normalize by MT-off reference
println("  Normalizing by MT-off reference (MTR)...")
mtr_volumes = Dict{String, Array{Float64,3}}()
for label in mt_labels
    mtr = volumes[label] ./ max.(mt_off, 1.0)  # avoid div by zero
    mtr_volumes[label] = mtr
end

# Compute voxelwise MTR for the strongest MT effect (closest offset, highest FA)
# flip-2_mt-7 at 1000 Hz with FA=628° should show the strongest MT effect
mtr_strongest = 1.0 .- mtr_volumes["flip-2_mt-7"]
mtr_strongest[.!mask_brain] .= 0
savenii(mtr_strongest, joinpath(OUTDIR, "MTR_strongest.nii.gz"); header=hdr)
@printf("  MTR range (brain): [%.3f, %.3f]\n",
        minimum(mtr_strongest[mask_brain]),
        maximum(mtr_strongest[mask_brain]))

# Compute mean MTR map across all offsets
mtr_mean = zeros(nx, ny, nz)
for label in mt_labels
    mtr_mean .+= (1.0 .- mtr_volumes[label])
end
mtr_mean ./= n_mt
mtr_mean[.!mask_brain] .= 0
savenii(mtr_mean, joinpath(OUTDIR, "MTR_mean.nii.gz"); header=hdr)

# ---------------------------------------------------------------------------
# 4. Voxelwise generalized Bloch fitting (single demo voxel)
# ---------------------------------------------------------------------------
println("\n[4/4] Demo: generalized Bloch fit on a single WM voxel...")
println("  NOTE: Full voxelwise fitting requires confirmed TRF (MT pulse duration).")
println("  Using estimated TRF = $(TRF_ESTIMATED*1000) ms.")

# Find a WM voxel with good signal (high MTR, inside mask)
candidates = findall(mask_brain .& (mtr_strongest .> 0.3) .& (mtr_strongest .< 0.7))
if !isempty(candidates)
    # Pick a central voxel
    mid_idx = candidates[length(candidates) ÷ 2]
    ix, iy, iz = Tuple(mid_idx)
    @printf("  WM voxel: (%d, %d, %d), MTR = %.3f\n", ix, iy, iz, mtr_strongest[ix, iy, iz])

    # Extract signal for this voxel across all MT conditions
    data_voxel = Float64[]
    α_voxel = Float64[]
    trf_voxel = Float64[]
    offset_voxel = Float64[]

    for (label, fa_deg, offset_hz) in QMT_PROTOCOL
        if label == "mt-off"
            continue
        end
        push!(data_voxel, mtr_volumes[label][ix, iy, iz])
        push!(α_voxel, deg2rad(fa_deg))
        push!(trf_voxel, TRF_ESTIMATED)
        push!(offset_voxel, Float64(offset_hz))
    end

    @printf("  Data vector: %d points\n", length(data_voxel))
    @printf("  Signal range: [%.4f, %.4f]\n", minimum(data_voxel), maximum(data_voxel))

    # The fit_gBloch function expects a specific train of RF pulses
    # For now, show the data is ready and what the next step would be
    println("\n  To perform full generalized Bloch fitting, need to confirm:")
    println("  1. MT pulse duration (TRF) — currently estimated at 12 ms")
    println("  2. MT pulse shape (Fermi, Gaussian, etc.)")
    println("  3. Whether the sequence uses balanced or spoiled gradients")
    println("  4. The readout flip angle (separate from MT pulse FA)")
    println("\n  These parameters are in the tfl_qMT_v09 sequence source but")
    println("  not in the BIDS sidecars. Contact CUBRIC for confirmation.")
else
    println("  No suitable WM voxels found for demo fit.")
end

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
println("\n" * "=" ^ 60)
println("Output files in $OUTDIR:")
println("=" ^ 60)
for f in readdir(OUTDIR)
    sz = filesize(joinpath(OUTDIR, f))
    @printf("  %-35s  %6.1f MB\n", f, sz / 1e6)
end
println("\nMTR maps computed. Full qMT parameter fitting awaits TRF confirmation.")
