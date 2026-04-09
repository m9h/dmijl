"""
    Differentiable MR-ARFI forward model.

End-to-end differentiable chain: intensity → radiation force → displacement → phase.
All operations use Zygote-compatible primitives (element-wise ops + FFTW).

The key insight: the entire chain is linear or element-wise, making
automatic differentiation straightforward:
- Radiation force: F = 2αI/c (element-wise)
- Spectral solve: u = IFFT(FFT(F)/(μk²)) (linear operator)
- Phase encoding: φ = γGδu (element-wise)

This enables gradient-based optimization of ultrasound parameters
(through the displacement field) and MRI parameters (through the
phase encoding).
"""

using Zygote, ForwardDiff

"""
    arfi_forward_differentiable(intensity, sound_speed, attenuation_np_m,
                                shear_modulus, grid_spacing, seq_params)
        -> (phase, displacement)

Differentiable end-to-end MR-ARFI forward model.

    intensity -> F = 2αI/c -> u = spectral_solve(F, μ) -> φ = γGδu

All operations are Zygote-compatible. The k-space grid (which depends
only on shape and spacing, not on data) is precomputed outside the
differentiable path.

# Arguments
- `intensity`: acoustic intensity (W/m^2)
- `sound_speed`: speed of sound (m/s)
- `attenuation_np_m`: absorption coefficient (Np/m)
- `shear_modulus`: tissue shear modulus (Pa, scalar or array)
- `grid_spacing`: voxel size (metres)
- `seq_params::ARFISequenceParams`: sequence parameters
"""
function arfi_forward_differentiable(
    intensity::AbstractArray{<:Real},
    sound_speed::AbstractArray{<:Real},
    attenuation_np_m::AbstractArray{<:Real},
    shear_modulus::Union{Real, AbstractArray{<:Real}},
    grid_spacing::Real,
    seq_params::ARFISequenceParams,
)
    # Step 1: Radiation force (element-wise, trivially differentiable)
    force = @. 2.0 * attenuation_np_m * intensity / sound_speed

    # Step 2: Spectral displacement solve
    # k_sq depends only on grid shape/spacing (not data), so exclude from AD
    k_sq = Zygote.@ignore _build_k_squared(size(force), grid_spacing)
    displacement = _solve_displacement_with_ksq(force, shear_modulus, k_sq)

    # Step 3: Phase encoding (linear in displacement)
    enc = arfi_encoding_coefficient(seq_params)
    phase = enc .* displacement

    return (phase, displacement)
end

"""
    _solve_displacement_with_ksq(force, shear_modulus, k_sq)

Inner displacement solve with pre-computed k_sq grid.
All operations are Zygote-safe (no fftfreq, no collect, no mutation).
"""
function _solve_displacement_with_ksq(
    force::AbstractArray{<:Real},
    shear_modulus::Union{Real, AbstractArray{<:Real}},
    k_sq::AbstractArray{<:Real},
)
    mu = if isa(shear_modulus, AbstractArray) && ndims(shear_modulus) > 0
        mean(shear_modulus)
    else
        Float64(shear_modulus)
    end
    mu = max(mu, 1e-6)

    F_hat = fft(Float64.(force))

    # Non-mutating k_sq masking for DC
    k_sq_safe = map(k -> k == 0.0 ? one(k) : k, k_sq)
    dc_mask = map(k -> k == 0.0 ? zero(eltype(F_hat)) : one(eltype(F_hat)), k_sq)

    u_hat = (F_hat .* dc_mask) ./ (mu .* k_sq_safe)
    return real.(ifft(u_hat))
end

"""
    arfi_phase_loss(intensity, sound_speed, attenuation_np_m,
                    shear_modulus, grid_spacing, seq_params,
                    target_indices) -> Float64

Loss function: negative squared phase at target voxels.
Minimizing this maximizes displacement sensitivity at the target.

Useful for optimizing ultrasound parameters (apodization, delays)
to maximize MR-ARFI signal at the focal region.
"""
function arfi_phase_loss(
    intensity::AbstractArray{<:Real},
    sound_speed::AbstractArray{<:Real},
    attenuation_np_m::AbstractArray{<:Real},
    shear_modulus::Union{Real, AbstractArray{<:Real}},
    grid_spacing::Real,
    seq_params::ARFISequenceParams,
    target_indices::AbstractVector{<:CartesianIndex},
)
    phase, _ = arfi_forward_differentiable(
        intensity, sound_speed, attenuation_np_m,
        shear_modulus, grid_spacing, seq_params,
    )
    target_phase = [phase[idx] for idx in target_indices]
    return -sum(target_phase .^ 2)
end

"""
    verify_arfi_gradient(; N=16, verbose=true) -> Bool

Verify that Zygote gradients through the ARFI chain match
finite-difference gradients from ForwardDiff.

Tests gradient of sum(phase^2) w.r.t. a scalar intensity scaling factor.
"""
function verify_arfi_gradient(; N::Int = 16, verbose::Bool = true)
    dx = 1e-3
    seq = ARFISequenceParams(msg_amplitude = 40e-3, msg_duration = 5e-3)

    # Fixed tissue properties
    c = fill(1560.0, N, N)
    alpha = fill(db_cm_to_neper_m(5.3), N, N)
    mu = 1.0e3
    base_intensity = zeros(N, N)
    base_intensity[N÷2, N÷2] = 1000.0

    # Function of scalar scaling parameter
    function loss_fn(scale)
        I = base_intensity .* scale
        phase, _ = arfi_forward_differentiable(I, c, alpha, mu, dx, seq)
        return sum(phase .^ 2)
    end

    # Zygote gradient
    g_zygote = Zygote.gradient(loss_fn, 1.0)[1]

    # Finite difference gradient
    g_fd = ForwardDiff.derivative(loss_fn, 1.0)

    if verbose
        println("ARFI gradient verification:")
        println("  Zygote:  $g_zygote")
        println("  ForwardDiff: $g_fd")
        if g_zygote !== nothing && g_fd != 0
            rel = abs(g_zygote - g_fd) / max(abs(g_fd), 1e-30)
            println("  Relative error: $(rel * 100)%")
        end
    end

    if g_zygote === nothing
        return false
    end
    return abs(g_zygote - g_fd) / max(abs(g_fd), 1e-30) < 0.01
end
