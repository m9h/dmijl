"""
    Phase processing pipeline using ROMEO + MriResearchTools.

Multi-echo magnitude + phase → unwrapped phase, B0 map, T2*, R2*, brain mask.
"""

using ROMEO
using MriResearchTools

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

"""
    load_multi_echo(mag_files, phase_files) -> (mag4d, pha4d, header)

Load multi-echo magnitude and phase NIfTI files into 4D Float32 arrays.
Phase is auto-rescaled from vendor encoding (e.g. Siemens 12-bit) to [-π, π].
Later echoes may have fewer slices; all are cropped to the minimum z-extent.
"""
function load_multi_echo(mag_files::Vector{String}, phase_files::Vector{String})
    length(mag_files) == length(phase_files) ||
        throw(ArgumentError("mag_files and phase_files must have same length"))
    n_echoes = length(mag_files)

    mag_vols = [readmag(f) for f in mag_files]
    pha_vols = [readphase(f) for f in phase_files]

    _to_f32(v) = Float32[v[i] for i in CartesianIndices(v)]

    nz = minimum(size(v, 3) for v in mag_vols)
    nx, ny = size(mag_vols[1], 1), size(mag_vols[1], 2)

    mag4d = zeros(Float32, nx, ny, nz, n_echoes)
    pha4d = zeros(Float32, nx, ny, nz, n_echoes)
    for e in 1:n_echoes
        mag4d[:,:,:,e] .= _to_f32(mag_vols[e])[:,:,1:nz]
        pha4d[:,:,:,e] .= _to_f32(pha_vols[e])[:,:,1:nz]
    end

    hdr = header(mag_vols[1])
    return mag4d, pha4d, hdr
end

# ---------------------------------------------------------------------------
# Processing steps
# ---------------------------------------------------------------------------

"""
    robust_brain_mask(mag3d) -> BitArray{3}

Compute a brain mask from a 3D magnitude volume using robust thresholding
(noise estimated from image corners) followed by morphological brain extraction.
"""
function robust_brain_mask(mag3d::AbstractArray{<:Real, 3})
    mask_raw = robustmask(mag3d)
    return brain_mask(mask_raw)
end

"""
    correct_bias_field(mag4d, mag_vol_echo1; nz=size(mag4d,3)) -> (mag4d_corrected, sensitivity)

Estimate the B1 receive sensitivity from the first-echo NIVolume (using voxel
sizes from the header) and divide it out of every echo. Returns the corrected
4D array and the 3D sensitivity map.
"""
function correct_bias_field(mag4d::Array{Float32, 4}, mag_vol_echo1; nz::Int=size(mag4d, 3))
    sensitivity_full = getsensitivity(mag_vol_echo1)
    sensitivity = sensitivity_full[:,:,1:nz]

    mag4d_corr = copy(mag4d)
    for e in 1:size(mag4d, 4)
        mag4d_corr[:,:,:,e] ./= sensitivity
    end
    mag4d_corr[.!isfinite.(mag4d_corr)] .= 0

    return mag4d_corr, sensitivity
end

"""
    unwrap_phase(phase4d, mag4d, TEs; mask=nothing) -> unwrapped

ROMEO multi-echo phase unwrapping with magnitude weighting.
`TEs` in ms. Returns unwrapped phase in radians.
"""
function unwrap_phase(phase4d::Array{Float32, 4}, mag4d::Array{Float32, 4},
                      TEs::Vector{Float64}; mask=nothing)
    kwargs = Dict{Symbol,Any}(:TEs => TEs, :mag => mag4d)
    if mask !== nothing
        kwargs[:mask] = mask
    end
    return romeo(phase4d; kwargs...)
end

"""
    voxel_quality(phase4d, mag4d, TEs) -> Array{Float32, 3}

Compute per-voxel ROMEO quality scores in [0, 1].
"""
function voxel_quality(phase4d::Array{Float32, 4}, mag4d::Array{Float32, 4},
                       TEs::Vector{Float64})
    return romeovoxelquality(phase4d; TEs=TEs, mag=mag4d)
end

"""
    compute_b0(unwrapped, mag4d, TEs; mask=nothing) -> Array{Float64}

B0 field map in Hz from unwrapped phase via SNR-weighted linear fit across echoes.
"""
function compute_b0(unwrapped::AbstractArray{<:Real, 4}, mag4d::AbstractArray{<:Real, 4},
                    TEs::Vector{Float64}; mask=nothing)
    b0 = calculateB0_unwrapped(unwrapped, mag4d, TEs)
    if mask !== nothing
        b0 .*= mask
    end
    return b0
end

"""
    compute_t2star(mag4d_corrected, TEs; mask=nothing, clamp_ms=200.0)
        -> (t2star, r2star)

T2* via NumART2star (trapezoidal integration, Hagberg et al. MRM 2002).
Returns T2* in ms and R2* in 1/s. Values outside the mask or above
`clamp_ms` are zeroed.
"""
function compute_t2star(mag4d_corr::AbstractArray{<:Real, 4}, TEs::Vector{Float64};
                        mask=nothing, clamp_ms::Float64=200.0)
    t2star = NumART2star(mag4d_corr, TEs)
    t2star = clamp.(t2star, 0, clamp_ms)
    if mask !== nothing
        t2star[.!mask] .= 0
    end

    r2star = r2s_from_t2s(t2star)
    r2star[.!isfinite.(r2star)] .= 0
    if mask !== nothing
        r2star[.!mask] .= 0
    end

    return t2star, r2star
end

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

"""
    PhaseResult

Holds all outputs from `process_phase`.
"""
struct PhaseResult
    mask::BitArray{3}
    sensitivity::Array{Float32, 3}
    unwrapped::Array{Float32, 4}
    quality::Array{Float32, 3}
    b0_hz::Array{Float64, 3}
    t2star_ms::Array{Float64, 3}
    r2star_s::Array{Float64, 3}
    header::Any
end

"""
    process_phase(mag_files, phase_files, TEs; kwargs...) -> PhaseResult

End-to-end multi-echo complex processing pipeline:

1. Load magnitude + phase NIfTI (vendor auto-rescaling)
2. Robust brain masking from first echo
3. B1 bias field correction (CLEAR-SWI)
4. ROMEO phase unwrapping (multi-echo temporal + spatial)
5. Voxel quality map
6. B0 field map (Hz)
7. T2*/R2* mapping (NumART2star)

# Arguments
- `mag_files::Vector{String}`: Paths to per-echo magnitude NIfTI files
- `phase_files::Vector{String}`: Paths to per-echo phase NIfTI files
- `TEs::Vector{Float64}`: Echo times in ms

# Keyword arguments
- `bias_correct::Bool=true`: Apply B1 bias field correction
- `clamp_t2star_ms::Float64=200.0`: Upper bound for T2* clamping
"""
function process_phase(mag_files::Vector{String}, phase_files::Vector{String},
                       TEs::Vector{Float64};
                       bias_correct::Bool=true,
                       clamp_t2star_ms::Float64=200.0)

    mag4d, pha4d, hdr = load_multi_echo(mag_files, phase_files)

    mask = robust_brain_mask(mag4d[:,:,:,1])

    if bias_correct
        # Need original NIVolume for voxel-size-aware sensitivity estimation
        mag_vol1 = readmag(mag_files[1])
        mag4d_corr, sensitivity = correct_bias_field(mag4d, mag_vol1; nz=size(mag4d, 3))
    else
        mag4d_corr = mag4d
        sensitivity = ones(Float32, size(mag4d)[1:3])
    end

    unwrapped = unwrap_phase(pha4d, mag4d, TEs; mask=mask)
    qual = voxel_quality(pha4d, mag4d, TEs)
    b0 = compute_b0(unwrapped, mag4d, TEs; mask=mask)
    t2s, r2s = compute_t2star(mag4d_corr, TEs; mask=mask, clamp_ms=clamp_t2star_ms)

    return PhaseResult(mask, sensitivity, unwrapped, Float32.(qual),
                       Float64.(b0), Float64.(t2s), Float64.(r2s), hdr)
end

"""
    save_phase_result(result::PhaseResult, outdir::String)

Write all maps from a PhaseResult to NIfTI files in `outdir`.
"""
function save_phase_result(result::PhaseResult, outdir::String)
    mkpath(outdir)
    hdr = result.header
    savenii(result.mask,           joinpath(outdir, "brain_mask.nii.gz"); header=hdr)
    savenii(result.sensitivity,    joinpath(outdir, "sensitivity_map.nii.gz"); header=hdr)
    savenii(result.unwrapped[:,:,:,1], joinpath(outdir, "unwrapped_echo1.nii.gz"); header=hdr)
    savenii(result.quality,        joinpath(outdir, "voxel_quality.nii.gz"); header=hdr)
    savenii(result.b0_hz,          joinpath(outdir, "B0_map_hz.nii.gz"); header=hdr)
    savenii(result.t2star_ms,      joinpath(outdir, "T2star.nii.gz"); header=hdr)
    savenii(result.r2star_s,       joinpath(outdir, "R2star.nii.gz"); header=hdr)
end
