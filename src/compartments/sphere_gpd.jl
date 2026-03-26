"""
SphereGPD — Gaussian Phase Distribution approximation for restricted diffusion
inside impermeable spheres.

Used as the soma compartment in SANDI (Palombo et al., NeuroImage 2020).
The signal is isotropic (direction-independent), depending only on sphere
diameter and intrinsic diffusivity.

Signal uses the Murday-Cotts/GPD expression with the first 20 roots of
d/dx[j_{3/2}(x)] = 0, where j_{3/2} is the spherical Bessel function.
"""

"""
    SPHERE_GPD_ROOTS

First 20 roots of d/dx[J_{3/2}(x)] = 0, i.e., values x_n where
the derivative of the spherical Bessel function j_{3/2}(x) vanishes.
These are the alpha_n * R values needed for the GPD approximation.
"""
const SPHERE_GPD_ROOTS = Float64[
    2.081575978, 5.940369990, 9.205840146,
    12.40444502, 15.57923641, 18.74264558,
    21.89969648, 25.05282528, 28.20336100,
    31.35209173, 34.49951492, 37.64596032,
    40.79165523, 43.93676147, 47.08139741,
    50.22565165, 53.36959180, 56.51327045,
    59.65672900, 62.79999505
]

"""Proton gyromagnetic ratio in rad/s/T."""
const GYROMAGNETIC_RATIO = 2.6752218744e8

Base.@kwdef struct SphereGPD <: AbstractCompartment
    diameter::Float64   # sphere diameter in metres
    D_intra::Float64    # intrinsic diffusivity in m^2/s
end

parameter_names(::SphereGPD) = (:diameter, :D_intra)
parameter_cardinality(::SphereGPD) = Dict(:diameter => 1, :D_intra => 1)
parameter_ranges(::SphereGPD) = Dict(
    :diameter => (1.0e-6, 30.0e-6),    # 1 to 30 micrometres
    :D_intra  => (0.0, 3.0e-9)         # up to 3 um^2/ms
)

_reconstruct(::SphereGPD, p::AbstractVector) = SphereGPD(diameter=p[1], D_intra=p[2])

"""
    signal(sphere::SphereGPD, acq::Acquisition)

Compute the GPD signal attenuation for restricted diffusion inside a sphere.

The acquisition must have `delta` and `Delta` set (not `nothing`).

Uses the Murday-Cotts expression:

    S = exp(-2 gamma^2 G^2 sum_n f(alpha_n))

where the sum runs over roots alpha_n of j'_{3/2}(alpha_n R) = 0, and

    f(alpha_n) = [alpha_n^(-4) * (alpha_n^2 R^2 - 2)^(-1)] *
                 [2 delta - (2 + exp(-a_n^2 D delta) - 2 exp(-a_n^2 D Delta)
                            - 2 exp(-a_n^2 D (Delta-delta)) + exp(-a_n^2 D (Delta+delta)))
                  / (a_n^2 D)]

and G is derived from b-value: b = gamma^2 G^2 delta^2 (Delta - delta/3),
so G^2 = b / [gamma^2 delta^2 (Delta - delta/3)].
"""
function signal(sphere::SphereGPD, acq::Acquisition)
    if acq.delta === nothing || acq.Delta === nothing
        throw(ArgumentError("SphereGPD requires delta and Delta in Acquisition"))
    end

    R = sphere.diameter / 2.0
    D = sphere.D_intra
    delta = acq.delta
    Delta = acq.Delta
    gamma = GYROMAGNETIC_RATIO
    n_meas = length(acq.bvalues)

    # When R is much larger than the diffusion length, the sphere is
    # effectively unrestricted and the signal converges to free diffusion.
    # The 20-root GPD sum cannot capture this limit, so fall back to Ball.
    diff_length = sqrt(2.0 * max(D, 0.0) * Delta)
    if R > 20.0 * diff_length
        return @. exp(-acq.bvalues * D)
    end

    result = ones(Float64, n_meas)

    # Precompute the GPD sum (independent of b-value / gradient strength)
    # The sum depends only on R, D, delta, Delta
    gpd_sum = _sphere_gpd_sum(R, D, delta, Delta)

    # For each measurement, compute G^2 from b-value, then the log-attenuation
    denom = gamma^2 * delta^2 * (Delta - delta / 3.0)

    for i in 1:n_meas
        b = acq.bvalues[i]
        if b <= 0.0 || denom <= 0.0
            result[i] = 1.0
            continue
        end
        # G^2 = b / (gamma^2 * delta^2 * (Delta - delta/3))
        G2 = b / denom
        # S = exp(-2 * gamma^2 * G^2 * gpd_sum)
        log_atten = -2.0 * gamma^2 * G2 * gpd_sum
        result[i] = exp(log_atten)
    end

    return result
end

"""
    _sphere_gpd_sum(R, D, delta, Delta)

Compute the GPD summation over the first N roots for a sphere of radius R.

Returns the value: sum_n [ alpha_n^{-4} (alpha_n^2 R^2 - 2)^{-1} *
    (2 delta - (2 + e^{-a^2 D delta} - 2 e^{-a^2 D Delta} - 2 e^{-a^2 D (Delta-delta)} + e^{-a^2 D (Delta+delta)}) / (a^2 D)) ]

where a = alpha_n (the root divided by R).
"""
function _sphere_gpd_sum(R::Float64, D::Float64, delta::Float64, Delta::Float64)
    # Handle edge case: R extremely small -> sum is ~0 (no phase dispersion)
    if R < 1e-10
        return 0.0
    end

    total = 0.0
    for xn in SPHERE_GPD_ROOTS
        # alpha_n = xn / R  (xn is the root alpha_n * R)
        alpha_n = xn / R
        a2 = alpha_n^2
        a2D = a2 * D

        # Denominator: alpha_n^4 * (alpha_n^2 * R^2 - 2)
        # alpha_n^2 * R^2 = xn^2, so (a2 * R^2 - 2) = (xn^2 - 2)
        denom_geom = a2^2 * (xn^2 - 2.0)

        # Skip if denominator is too small (shouldn't happen for standard roots, xn > 2)
        if abs(denom_geom) < 1e-30
            continue
        end

        # Time-dependent numerator
        # 2*delta - [2 + exp(-a2D*delta) - 2*exp(-a2D*Delta) - 2*exp(-a2D*(Delta-delta)) + exp(-a2D*(Delta+delta))] / (a2D)
        if a2D < 1e-20
            # For very small a2D, use Taylor expansion: numerator -> 0
            continue
        end

        bracket = 2.0 + exp(-a2D * delta) - 2.0 * exp(-a2D * Delta) -
                  2.0 * exp(-a2D * (Delta - delta)) + exp(-a2D * (Delta + delta))
        numer = 2.0 * delta - bracket / a2D

        total += numer / denom_geom
    end

    return total
end
