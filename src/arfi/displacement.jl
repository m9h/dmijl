"""
    Spectral Green's function solver for quasi-static tissue displacement.

Solves the Poisson-like equation mu * nabla^2 u = -F in Fourier space,
where F is the radiation force density and mu is the tissue shear modulus.

The spectral solution is:
    u_hat(k) = F_hat(k) / (mu * |k|^2)
    u = IFFT(u_hat)

Fully differentiable via FFTW's AbstractFFTs Zygote rules.

References:
    - Sarvazyan et al. (1998). Shear wave elasticity imaging. UMB 24(9).
    - Greenleaf et al. (2003). Radiation force in tissue. UMB 29(7).
"""

using FFTW

"""
    solve_displacement_spectral(force, shear_modulus, grid_spacing) -> Array{Float64}

Solve for quasi-static tissue displacement from radiation force via
spectral Green's function (FFT-based).

    mu * nabla^2 u = -F
    =>  u_hat(k) = F_hat(k) / (mu * |k|^2)

For heterogeneous shear modulus (array), uses the spatial mean as a
first-order approximation (iterative refinement deferred to Phase 4).

DC component (k=0) is zeroed to enforce no net displacement.

# Arguments
- `force`: radiation force density (N/m^3), any spatial shape (2D or 3D)
- `shear_modulus`: shear modulus in Pa (scalar or array matching force shape)
- `grid_spacing`: uniform grid spacing (metres)

# Returns
Displacement field (metres), same shape as force.
"""
function solve_displacement_spectral(
    force::AbstractArray{<:Real},
    shear_modulus::Union{Real, AbstractArray{<:Real}},
    grid_spacing::Real,
)
    ndim = ndims(force)

    # Use mean shear modulus for spectral solve
    mu = if isa(shear_modulus, AbstractArray) && ndims(shear_modulus) > 0
        mean(shear_modulus)
    else
        Float64(shear_modulus)
    end
    mu = max(mu, 1e-6)

    # FFT of force field
    F_hat = fft(Float64.(force))

    # Build |k|^2 grid in Fourier space
    k_sq = _build_k_squared(size(force), grid_spacing)

    # Avoid division by zero at DC (k=0) — non-mutating for Zygote
    k_sq_safe = map(k -> k == 0.0 ? one(k) : k, k_sq)

    # Spectral Green's function, with DC zeroed (no net displacement)
    # Use element-wise mask instead of setindex! for Zygote compatibility
    dc_mask = map(k -> k == 0.0 ? zero(eltype(F_hat)) : one(eltype(F_hat)), k_sq)
    u_hat = (F_hat .* dc_mask) ./ (mu .* k_sq_safe)

    # IFFT back to real space
    return real.(ifft(u_hat))
end

"""
    _build_k_squared(shape, grid_spacing) -> Array{Float64}

Build the |k|^2 grid in Fourier space for an N-dimensional domain.
Uses FFTW frequency conventions (fftfreq).
Non-mutating implementation for Zygote compatibility.
"""
function _build_k_squared(shape::Tuple, grid_spacing::Real)
    ndim = length(shape)

    # Build each dimension's k_i^2 contribution and sum (non-mutating)
    # collect() converts Frequencies type to plain Vector for Zygote compatibility
    components = map(1:ndim) do d
        n = shape[d]
        freqs = collect(Float64, fftfreq(n, 1.0 / grid_spacing))
        ki_sq = (2.0 * pi .* freqs) .^ 2
        reshape_dims = ntuple(i -> i == d ? n : 1, ndim)
        reshape(ki_sq, reshape_dims)
    end

    return reduce(.+, components)
end
