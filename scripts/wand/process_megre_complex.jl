#!/usr/bin/env julia
# WAND sub-08033 ses-06: Complex-valued MEGRE processing with ROMEO + MriResearchTools
#
# Demonstrates the full pipeline on 7T ASPIRE multi-echo GRE data:
#   1. Load 7-echo magnitude + phase (Siemens 12-bit → [-π, π])
#   2. Robust brain masking from magnitude
#   3. Bias field correction (CLEAR-SWI method)
#   4. ROMEO phase unwrapping (multi-echo temporal + spatial)
#   5. B0 field map estimation from unwrapped phase
#   6. T2* mapping (NumART2star)
#   7. Voxel quality map for QSM-ready masking
#
# Data: 7T Siemens, 0.67mm iso, 32ch coil, ASPIRE phase combination
# TEs: [5, 10, 15, 20, 25, 30, 35] ms

using Pkg
Pkg.activate(@__DIR__)

# Install if not present
for p in ["ROMEO", "MriResearchTools", "NIfTI", "Statistics"]
    try
        @eval using $(Symbol(p))
    catch
        Pkg.add(p)
        @eval using $(Symbol(p))
    end
end

using ROMEO
using MriResearchTools
using NIfTI
using Statistics
using Printf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const SUB = "sub-08033"
const SES = "ses-06"
const RAWDIR = "/data/raw/wand/$SUB/$SES/anat"
const OUTDIR = "/data/datasets/wand/derivatives/romeo/$SUB/$SES"
const REFDIR = "/data/raw/wand/derivatives/qmri/$SUB/$SES"  # existing T2*/R2* maps

const N_ECHOES = 7
const TEs = Float64[5, 10, 15, 20, 25, 30, 35]  # ms (from BIDS JSON)

mkpath(OUTDIR)

# ---------------------------------------------------------------------------
# 1. Load multi-echo magnitude and phase
# ---------------------------------------------------------------------------
println("=" ^ 60)
println("WAND MEGRE Complex Processing: $SUB $SES")
println("=" ^ 60)

println("\n[1/7] Loading $N_ECHOES-echo magnitude + phase data...")

mag_files = [joinpath(RAWDIR, "$(SUB)_$(SES)_echo-$(lpad(e,2,'0'))_part-mag_MEGRE.nii.gz")
             for e in 1:N_ECHOES]
pha_files = [joinpath(RAWDIR, "$(SUB)_$(SES)_echo-$(lpad(e,2,'0'))_part-phase_MEGRE.nii.gz")
             for e in 1:N_ECHOES]

# readmag / readphase from MriResearchTools handle vendor quirks
# readphase auto-rescales Siemens 12-bit [0,4095] → [-π, π]
mag_vols = [readmag(f) for f in mag_files]
pha_vols = [readphase(f) for f in pha_files]

# Stack into 4D arrays (x, y, z, echo)
# NIVolume doesn't support direct broadcast; collect each echo to plain Array first
to_f32(v) = Float32[v[i] for i in CartesianIndices(v)]

# Later echoes may have fewer slices at 7T — crop to minimum
nz_min = minimum(size(v, 3) for v in mag_vols)
nx, ny = size(mag_vols[1], 1), size(mag_vols[1], 2)
nz = nz_min

mag4d = zeros(Float32, nx, ny, nz, N_ECHOES)
pha4d = zeros(Float32, nx, ny, nz, N_ECHOES)
for e in 1:N_ECHOES
    m = to_f32(mag_vols[e])
    p = to_f32(pha_vols[e])
    mag4d[:,:,:,e] .= m[:,:,1:nz]
    pha4d[:,:,:,e] .= p[:,:,1:nz]
end

hdr = header(mag_vols[1])
@printf("  Dimensions: %d × %d × %d × %d echoes\n", nx, ny, nz, N_ECHOES)
@printf("  Phase range: [%.2f, %.2f] rad\n", minimum(pha4d), maximum(pha4d))
@printf("  Magnitude range: [%.1f, %.1f]\n", minimum(mag4d), maximum(mag4d))

# ---------------------------------------------------------------------------
# 2. Robust brain masking from first-echo magnitude
# ---------------------------------------------------------------------------
println("\n[2/7] Computing robust brain mask...")

# robustmask: automatic threshold from noise distribution in image corners
mask_raw = robustmask(mag4d[:,:,:,1])
mask_brain = brain_mask(mask_raw)

n_voxels = sum(mask_brain)
n_total = length(mask_brain)
@printf("  Brain voxels: %d / %d (%.1f%%)\n", n_voxels, n_total, 100 * n_voxels / n_total)

savenii(mask_brain, joinpath(OUTDIR, "brain_mask.nii.gz"); header=hdr)
println("  Saved: brain_mask.nii.gz")

# ---------------------------------------------------------------------------
# 3. Bias field correction (CLEAR-SWI / boxsegment method)
# ---------------------------------------------------------------------------
println("\n[3/7] Correcting B1 bias field (CLEAR-SWI method)...")

# makehomogeneous estimates and divides out the receive sensitivity profile
# Uses first echo for field estimation, applies to all echoes
mag4d_corr = copy(mag4d)
sensitivity_full = getsensitivity(mag_vols[1])  # NIVolume overload uses voxel sizes from header
sensitivity = sensitivity_full[:,:,1:nz]  # crop to match echo-cropped volume
for e in 1:N_ECHOES
    mag4d_corr[:,:,:,e] ./= sensitivity
end

# Replace NaN/Inf from division
mag4d_corr[.!isfinite.(mag4d_corr)] .= 0

savenii(sensitivity, joinpath(OUTDIR, "sensitivity_map.nii.gz"); header=hdr)
savenii(mag4d_corr[:,:,:,1], joinpath(OUTDIR, "echo1_mag_homogeneous.nii.gz"); header=hdr)
println("  Saved: sensitivity_map.nii.gz, echo1_mag_homogeneous.nii.gz")

# ---------------------------------------------------------------------------
# 4. ROMEO phase unwrapping
# ---------------------------------------------------------------------------
println("\n[4/7] ROMEO phase unwrapping (multi-echo, magnitude-weighted)...")

# ROMEO uses temporal coherence across echoes + magnitude weighting
# TEs in ms, magnitude for path quality, mask to restrict computation
unwrapped = romeo(pha4d; TEs=TEs, mag=mag4d, mask=mask_brain)

@printf("  Unwrapped range: [%.2f, %.2f] rad\n", minimum(unwrapped), maximum(unwrapped))

savenii(unwrapped[:,:,:,1], joinpath(OUTDIR, "unwrapped_echo1.nii.gz"); header=hdr)
println("  Saved: unwrapped_echo1.nii.gz")

# ---------------------------------------------------------------------------
# 5. Voxel quality map
# ---------------------------------------------------------------------------
println("\n[5/7] Computing ROMEO voxel quality map...")

# Per-voxel reliability score in [0, 1] — useful for downstream QSM masking
quality = romeovoxelquality(pha4d; TEs=TEs, mag=mag4d)

savenii(quality, joinpath(OUTDIR, "voxel_quality.nii.gz"); header=hdr)
@printf("  Quality range: [%.3f, %.3f]\n", minimum(quality), maximum(quality))
println("  Saved: voxel_quality.nii.gz")

# ---------------------------------------------------------------------------
# 6. B0 field map from unwrapped phase
# ---------------------------------------------------------------------------
println("\n[6/7] Computing B0 field map from unwrapped phase...")

# calculateB0_unwrapped: weighted linear fit across echoes → Hz
B0_hz = calculateB0_unwrapped(unwrapped, mag4d, TEs)

# Apply mask
B0_hz .*= mask_brain

savenii(B0_hz, joinpath(OUTDIR, "B0_map_hz.nii.gz"); header=hdr)
@printf("  B0 range (brain): [%.1f, %.1f] Hz\n",
        minimum(B0_hz[mask_brain]), maximum(B0_hz[mask_brain]))
println("  Saved: B0_map_hz.nii.gz")

# ---------------------------------------------------------------------------
# 7. T2* mapping (NumART2star)
# ---------------------------------------------------------------------------
println("\n[7/7] Computing T2* map (NumART2star)...")

# NumART2star: trapezoidal integration method (Hagberg et al. MRM 2002)
# Works on magnitude data, returns T2* in same units as TEs (ms here)
t2star = NumART2star(mag4d_corr, TEs)

# Clamp to physiological range and mask
t2star[.!mask_brain] .= 0
t2star = clamp.(t2star, 0, 200)  # ms, reasonable for 7T brain

savenii(t2star, joinpath(OUTDIR, "T2star_romeo.nii.gz"); header=hdr)

# R2* = 1000 / T2* (convert ms → 1/s)
r2star = r2s_from_t2s(t2star)
r2star[.!isfinite.(r2star)] .= 0
r2star[.!mask_brain] .= 0

savenii(r2star, joinpath(OUTDIR, "R2star_romeo.nii.gz"); header=hdr)

@printf("  T2* range (brain): [%.2f, %.2f] ms\n",
        minimum(t2star[mask_brain .& (t2star .> 0)]),
        maximum(t2star[mask_brain .& (t2star .> 0)]))
@printf("  R2* range (brain): [%.1f, %.1f] 1/s\n",
        minimum(r2star[mask_brain .& (r2star .> 0)]),
        maximum(r2star[mask_brain .& (r2star .> 0)]))
println("  Saved: T2star_romeo.nii.gz, R2star_romeo.nii.gz")

# ---------------------------------------------------------------------------
# Validation: compare against existing qMRI derivatives
# ---------------------------------------------------------------------------
println("\n" * "=" ^ 60)
println("Validation against existing qMRI derivatives")
println("=" ^ 60)

ref_t2s_path = joinpath(REFDIR, "T2star_map.nii.gz")
ref_r2s_path = joinpath(REFDIR, "R2star_map.nii.gz")

if isfile(ref_t2s_path) && isfile(ref_r2s_path)
    ref_t2s = Float64.(niread(ref_t2s_path))
    ref_r2s = Float64.(niread(ref_r2s_path))

    # Reference T2* is in seconds (range 0-0.2), ours is in ms
    # Reference R2* is in 1/s (range 0-500), same as ours
    ref_t2s_ms = ref_t2s .* 1000  # s → ms

    # Spatial dimensions may differ slightly (336×336×90 vs 92)
    nz_ref = size(ref_t2s, 3)
    nz_min = min(nz, nz_ref)

    # Compare in overlapping brain region
    mask_overlap = mask_brain[:,:,1:nz_min]
    our_t2s = t2star[:,:,1:nz_min]
    their_t2s = ref_t2s_ms[:,:,1:nz_min]

    valid = mask_overlap .& (our_t2s .> 0) .& (their_t2s .> 0) .&
            (our_t2s .< 200) .& (their_t2s .< 200)

    if sum(valid) > 0
        diff = our_t2s[valid] .- their_t2s[valid]
        @printf("  Overlap voxels: %d\n", sum(valid))
        @printf("  T2* mean ours:   %.2f ms\n", mean(our_t2s[valid]))
        @printf("  T2* mean theirs: %.2f ms\n", mean(their_t2s[valid]))
        @printf("  Mean difference: %.2f ms\n", mean(diff))
        @printf("  Correlation:     %.4f\n",
                cor(our_t2s[valid][:], their_t2s[valid][:]))
    end
else
    println("  Reference maps not found, skipping validation.")
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
println("\nDone.")
