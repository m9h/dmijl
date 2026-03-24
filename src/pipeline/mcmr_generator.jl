"""
Monte Carlo MR training data generator using MCMRSimulator.jl.

Generates (params, signals) training pairs by running Monte Carlo random walks
through realistic tissue microstructure geometries (packed cylinders for white
matter axons, packed spheres for grey matter soma).

This module bridges MCMRSimulator.jl's physics engine with the surrogate
training pipeline in Microstructure.jl. The `mcmr_data_fn` convenience
function returns a closure compatible with `train_surrogate!`.

References:
- MCMRSimulator.jl: https://git.fmrib.ox.ac.uk/ndcn0236/MCMRSimulator.jl
- SANDI model (spheres for soma): Palombo et al., NeuroImage 2020
"""

using Random
using Statistics
using MRIBuilder: Sequence, DWI, build_sequence

# ------------------------------------------------------------------ #
# Geometry specification
# ------------------------------------------------------------------ #

"""
    MCMRGeometry

Parameterizes tissue microstructure for Monte Carlo simulation.

# Fields
- `geometry_type`: `:cylinders` (white matter axons) or `:spheres` (grey matter soma)
- `mean_radius`: mean radius of structures in um (typical: 0.5-5.0 for axons, 3-8 for soma)
- `radius_variance`: variance of the Gamma-distributed radii in um^2
- `volume_fraction`: target packing density (0.0-0.85)
- `box_size`: repeating unit cell size in um (e.g., 20.0)
- `rotation`: 3x3 rotation matrix for cylinder orientation (ignored for spheres)
"""
struct MCMRGeometry
    geometry_type::Symbol
    mean_radius::Float64
    radius_variance::Float64
    volume_fraction::Float64
    box_size::Float64
    rotation::Union{Nothing, Matrix{Float64}}
end

function MCMRGeometry(;
    geometry_type::Symbol = :cylinders,
    mean_radius::Float64 = 1.0,
    radius_variance::Float64 = 0.1,
    volume_fraction::Float64 = 0.6,
    box_size::Float64 = 20.0,
    rotation = nothing,
)
    @assert geometry_type in (:cylinders, :spheres) "geometry_type must be :cylinders or :spheres"
    @assert 0.0 < mean_radius "mean_radius must be positive"
    @assert 0.0 <= radius_variance "radius_variance must be non-negative"
    @assert 0.0 < volume_fraction <= 0.85 "volume_fraction must be in (0, 0.85]"
    @assert 0.0 < box_size "box_size must be positive"
    MCMRGeometry(geometry_type, mean_radius, radius_variance, volume_fraction,
                 box_size, rotation)
end

"""
    param_names(geo::MCMRGeometry)

Return the parameter names for this geometry type.
"""
function param_names(geo::MCMRGeometry)
    if geo.geometry_type == :cylinders
        return ["mean_radius", "radius_variance", "volume_fraction"]
    else
        return ["mean_radius", "radius_variance", "volume_fraction"]
    end
end

param_dim(::MCMRGeometry) = 3  # (mean_radius, radius_variance, volume_fraction)

# ------------------------------------------------------------------ #
# Geometry samplers
# ------------------------------------------------------------------ #

"""
    sample_cylinder_geometry(rng) -> (params_vector, MCMRGeometry)

Sample a random white matter geometry: parallel cylinders with varying
radii and packing density. Returns a 3-element parameter vector
[mean_radius, radius_variance, volume_fraction] and the corresponding
MCMRGeometry.

Priors:
- mean_radius ~ Uniform(0.3, 5.0) um  (axon radii in white matter)
- radius_variance ~ Uniform(0.01, 0.5) um^2
- volume_fraction ~ Uniform(0.3, 0.8)
"""
function sample_cylinder_geometry(rng::AbstractRNG)
    mean_radius = 0.3 + rand(rng) * 4.7       # [0.3, 5.0] um
    radius_variance = 0.01 + rand(rng) * 0.49  # [0.01, 0.5] um^2
    volume_fraction = 0.3 + rand(rng) * 0.5    # [0.3, 0.8]

    params = Float32[mean_radius, radius_variance, volume_fraction]

    geo = MCMRGeometry(
        geometry_type = :cylinders,
        mean_radius = mean_radius,
        radius_variance = radius_variance,
        volume_fraction = volume_fraction,
    )

    return params, geo
end

"""
    sample_sphere_geometry(rng) -> (params_vector, MCMRGeometry)

Sample a random grey matter (SANDI-like) geometry: packed spheres with
varying radii and density. Returns a 3-element parameter vector
[mean_radius, radius_variance, volume_fraction] and the corresponding
MCMRGeometry.

Priors:
- mean_radius ~ Uniform(3.0, 8.0) um  (soma radii)
- radius_variance ~ Uniform(0.1, 2.0) um^2
- volume_fraction ~ Uniform(0.2, 0.7)
"""
function sample_sphere_geometry(rng::AbstractRNG)
    mean_radius = 3.0 + rand(rng) * 5.0       # [3.0, 8.0] um
    radius_variance = 0.1 + rand(rng) * 1.9    # [0.1, 2.0] um^2
    volume_fraction = 0.2 + rand(rng) * 0.5    # [0.2, 0.7]

    params = Float32[mean_radius, radius_variance, volume_fraction]

    geo = MCMRGeometry(
        geometry_type = :spheres,
        mean_radius = mean_radius,
        radius_variance = radius_variance,
        volume_fraction = volume_fraction,
    )

    return params, geo
end

# ------------------------------------------------------------------ #
# MCMRSimulator geometry construction
# ------------------------------------------------------------------ #

"""
    build_mcmr_geometry(geo::MCMRGeometry)

Construct the MCMRSimulator geometry objects (Cylinders or Spheres)
from an MCMRGeometry specification.

Uses `random_positions_radii` to generate packed, non-overlapping
configurations within a repeating unit cell.
"""
function build_mcmr_geometry(geo::MCMRGeometry)
    ndim = geo.geometry_type == :cylinders ? 2 : 3
    box = geo.geometry_type == :cylinders ?
        (geo.box_size, geo.box_size) :
        (geo.box_size, geo.box_size, geo.box_size)

    # Generate random positions and radii via MCMRSimulator
    positions, radii = MCMRSimulator.random_positions_radii(
        box, geo.volume_fraction, ndim;
        mean = geo.mean_radius,
        variance = geo.radius_variance,
        min_radius = max(0.1, geo.mean_radius * 0.1),
        max_radius = geo.mean_radius * 3.0,
    )

    if geo.geometry_type == :cylinders
        repeats = (geo.box_size, geo.box_size)
        kwargs = Dict{Symbol, Any}(
            :radius => radii,
            :position => positions,
            :repeats => repeats,
        )
        if geo.rotation !== nothing
            kwargs[:rotation] = geo.rotation
        end
        return MCMRSimulator.Cylinders(; kwargs...)
    else
        repeats = (geo.box_size, geo.box_size, geo.box_size)
        return MCMRSimulator.Spheres(
            radius = radii,
            position = positions,
            repeats = repeats,
        )
    end
end

# ------------------------------------------------------------------ #
# Signal extraction helpers
# ------------------------------------------------------------------ #

"""
    extract_signal(readout_result) -> Vector{Float64}

Extract the transverse magnetization magnitude from an MCMRSimulator
readout result (SpinOrientationSum). Returns the signal normalized
by the number of spins.
"""
function extract_signal(result, n_spins::Int)
    # readout() returns SpinOrientationSum when return_snapshot=false.
    # transverse() gives the magnitude of the transverse component.
    sig = MCMRSimulator.transverse(result) / n_spins
    return sig
end

# ------------------------------------------------------------------ #
# Main data generation function
# ------------------------------------------------------------------ #

"""
    generate_mcmr_training_data(
        geometry_sampler, sequence, n_samples;
        n_spins=10_000, diffusivity=2.0, R1=1e-3, R2=0.015,
        skip_TR=2, bounding_box=500,
    )

Generate (params, signals) training data pairs using MCMRSimulator.jl
for restricted diffusion in realistic tissue geometries.

# Arguments
- `geometry_sampler`: function `(rng) -> (params_vec, MCMRGeometry)` that
  samples random microstructure configurations. Use `sample_cylinder_geometry`
  for white matter or `sample_sphere_geometry` for grey matter.
- `sequence`: an `MRIBuilder.Sequence` defining the dMRI pulse sequence
  (e.g., `DWI(bval=1., TE=80, TR=1000)`).
- `n_samples`: number of (geometry, signal) training pairs to generate.

# Keyword arguments
- `n_spins`: number of Monte Carlo spins per simulation (default: 10,000).
  More spins → less noisy signals but slower.
- `diffusivity`: free water diffusivity in um^2/ms (default: 2.0).
- `R1`: longitudinal relaxation rate in kHz (default: 1e-3, i.e. T1=1000ms).
- `R2`: transverse relaxation rate in kHz (default: 0.015, i.e. T2~67ms).
- `skip_TR`: number of TRs to skip for magnetization equilibration (default: 2).
- `bounding_box`: size of the simulation voxel in um (default: 500).
- `verbose`: print progress messages (default: true).

# Returns
- `params::Matrix{Float32}`: microstructure parameters, shape (param_dim, n_samples)
- `signals::Matrix{Float32}`: normalized dMRI signals, shape (signal_dim, n_samples)

Each column of `params` contains [mean_radius, radius_variance, volume_fraction].
Each column of `signals` contains the b0-normalized transverse signal for each
sequence measurement (if multiple sequences are provided, they are concatenated).
"""
function generate_mcmr_training_data(
    geometry_sampler,
    sequence,
    n_samples::Int;
    n_spins::Int = 10_000,
    diffusivity::Float64 = 2.0,
    R1::Float64 = 1e-3,
    R2::Float64 = 0.015,
    skip_TR::Int = 2,
    bounding_box::Int = 500,
    verbose::Bool = true,
)
    rng = Random.default_rng()

    # Storage — allocated lazily after first sample reveals signal dimension
    params = nothing
    signals = nothing

    for j in 1:n_samples
        # 1. Sample random microstructure parameters
        param_vec, geo = geometry_sampler(rng)

        # 2. Build MCMRSimulator geometry
        mcmr_geometry = build_mcmr_geometry(geo)

        # 3. Create Simulation
        sim = MCMRSimulator.Simulation(
            sequence;
            geometry = mcmr_geometry,
            diffusivity = diffusivity,
            R1 = R1,
            R2 = R2,
            verbose = false,
        )

        # 4. Run Monte Carlo simulation
        result = MCMRSimulator.readout(n_spins, sim; skip_TR = skip_TR, bounding_box = bounding_box)

        # 5. Extract signal
        sig = extract_signal(result, n_spins)

        # Allocate on first iteration once we know signal dimension
        if params === nothing
            param_dim_val = length(param_vec)
            # For a single sequence, sig is a scalar; for multi-readout it
            # may be a vector. We handle both cases.
            signal_vec = sig isa Number ? Float32[sig] : Float32.(collect(sig))
            signal_dim_val = length(signal_vec)

            params = zeros(Float32, param_dim_val, n_samples)
            signals = zeros(Float32, signal_dim_val, n_samples)

            params[:, 1] = param_vec
            signals[:, 1] = signal_vec
        else
            signal_vec = sig isa Number ? Float32[sig] : Float32.(collect(sig))
            params[:, j] = param_vec
            signals[:, j] = signal_vec
        end

        if verbose && (j % max(1, n_samples ÷ 10) == 0 || j == 1)
            @info "[MCMR] Generated $j/$n_samples samples"
        end
    end

    return params, signals
end

# ------------------------------------------------------------------ #
# Multi-sequence version
# ------------------------------------------------------------------ #

"""
    generate_mcmr_training_data(
        geometry_sampler, sequences::Vector, n_samples; kwargs...
    )

Generate training data across multiple sequences. The signals from each
sequence are concatenated along the first dimension, producing a signal
vector of length sum(signal_dim_per_sequence).
"""
function generate_mcmr_training_data(
    geometry_sampler,
    sequences::AbstractVector{<:Sequence},
    n_samples::Int;
    n_spins::Int = 10_000,
    diffusivity::Float64 = 2.0,
    R1::Float64 = 1e-3,
    R2::Float64 = 0.015,
    skip_TR::Int = 2,
    bounding_box::Int = 500,
    verbose::Bool = true,
)
    rng = Random.default_rng()

    params = nothing
    signals = nothing

    for j in 1:n_samples
        param_vec, geo = geometry_sampler(rng)
        mcmr_geometry = build_mcmr_geometry(geo)

        sim = MCMRSimulator.Simulation(
            sequences;
            geometry = mcmr_geometry,
            diffusivity = diffusivity,
            R1 = R1,
            R2 = R2,
            verbose = false,
        )

        result = MCMRSimulator.readout(n_spins, sim; skip_TR = skip_TR, bounding_box = bounding_box)

        # For multiple sequences, result is a vector of SpinOrientationSum.
        # Extract and concatenate signals across sequences.
        sig_parts = [extract_signal(r, n_spins) for r in result]
        # Each part may be scalar or vector; flatten into a single vector.
        signal_vec = Float32.(vcat([s isa Number ? [s] : collect(s) for s in sig_parts]...))

        if params === nothing
            param_dim_val = length(param_vec)
            signal_dim_val = length(signal_vec)
            params = zeros(Float32, param_dim_val, n_samples)
            signals = zeros(Float32, signal_dim_val, n_samples)
        end

        params[:, j] = param_vec
        signals[:, j] = signal_vec

        if verbose && (j % max(1, n_samples ÷ 10) == 0 || j == 1)
            @info "[MCMR] Generated $j/$n_samples samples (multi-seq)"
        end
    end

    return params, signals
end

# ------------------------------------------------------------------ #
# Convenience closure for train_surrogate! interface
# ------------------------------------------------------------------ #

"""
    mcmr_data_fn(sequence, geometry_type; n_spins=10_000, kwargs...)

Return a closure `(rng, n) -> (params, signals)` compatible with the
`train_surrogate!` interface.

# Arguments
- `sequence`: an `MRIBuilder.Sequence` or vector of sequences.
- `geometry_type`: `:cylinders` for white matter or `:spheres` for grey matter.

# Keyword arguments
- `n_spins`: Monte Carlo spins per simulation (default: 10,000).
- `diffusivity`: free water diffusivity in um^2/ms (default: 2.0).
- `R1`: longitudinal relaxation rate in kHz (default: 1e-3).
- `R2`: transverse relaxation rate in kHz (default: 0.015).
- `skip_TR`: TRs to skip for equilibration (default: 2).
- `bounding_box`: simulation voxel size in um (default: 500).

# Example
```julia
using MRIBuilder
seq = DWI(bval=1., TE=80, TR=1000)
data_fn = mcmr_data_fn(seq, :cylinders; n_spins=5000)

# Use with train_surrogate!
model = build_surrogate(param_dim=3, signal_dim=1)
ps, st = Lux.setup(rng, model)
ps, st, losses = train_surrogate!(model, ps, st, data_fn; n_steps=1000)
```
"""
function mcmr_data_fn(
    sequence,
    geometry_type::Symbol;
    n_spins::Int = 10_000,
    diffusivity::Float64 = 2.0,
    R1::Float64 = 1e-3,
    R2::Float64 = 0.015,
    skip_TR::Int = 2,
    bounding_box::Int = 500,
)
    # Select the appropriate geometry sampler
    sampler = if geometry_type == :cylinders
        sample_cylinder_geometry
    elseif geometry_type == :spheres
        sample_sphere_geometry
    else
        error("Unknown geometry_type: $geometry_type. Use :cylinders or :spheres.")
    end

    # Return closure matching the (rng, n) -> (params, signals) interface
    return function (rng::AbstractRNG, n::Int)
        params = nothing
        signals = nothing

        for j in 1:n
            param_vec, geo = sampler(rng)
            mcmr_geometry = build_mcmr_geometry(geo)

            sim = MCMRSimulator.Simulation(
                sequence;
                geometry = mcmr_geometry,
                diffusivity = diffusivity,
                R1 = R1,
                R2 = R2,
                verbose = false,
            )

            result = MCMRSimulator.readout(
                n_spins, sim;
                skip_TR = skip_TR,
                bounding_box = bounding_box,
            )

            # Handle single vs multi-sequence results
            if result isa AbstractArray
                sig_parts = [extract_signal(r, n_spins) for r in result]
                signal_vec = Float32.(vcat([s isa Number ? [s] : collect(s) for s in sig_parts]...))
            else
                sig = extract_signal(result, n_spins)
                signal_vec = sig isa Number ? Float32[sig] : Float32.(collect(sig))
            end

            if params === nothing
                params = zeros(Float32, length(param_vec), n)
                signals = zeros(Float32, length(signal_vec), n)
            end

            params[:, j] = param_vec
            signals[:, j] = signal_vec
        end

        return params, signals
    end
end
