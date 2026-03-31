"""
PlaneCallaghan — Gaussian Phase Distribution approximation for restricted
diffusion between two impermeable parallel planes.

Completes the geometry set: cylinders (RestrictedCylinder), spheres (SphereGPD),
and planes (PlaneCallaghan). The signal depends on the plane separation (a),
intrinsic diffusivity, and the angle between gradient and plane normal.

Signal uses the Callaghan (1997) GPD expression with cosine basis functions.

Reference: Callaghan, P.T. (1997) "A simple matrix formalism for spin echo
analysis of restricted diffusion under generalized gradient waveforms."
J. Magn. Reson., 129, 74-84.
"""

Base.@kwdef struct PlaneCallaghan <: AbstractCompartment
    a::Float64      # plane separation in metres
    D_intra::Float64  # intrinsic diffusivity in m^2/s
    normal::Vector{Float64} = [0.0, 0.0, 1.0]  # plane normal direction
end

parameter_names(::PlaneCallaghan) = (:a, :D_intra, :normal)
parameter_cardinality(::PlaneCallaghan) = Dict(:a => 1, :D_intra => 1, :normal => 3)
parameter_ranges(::PlaneCallaghan) = Dict(
    :a => (1e-7, 50e-6),       # 0.1 - 50 um
    :D_intra => (0.0, 3.0e-9), # m^2/s
    :normal => (-1.0, 1.0),
)

_reconstruct(::PlaneCallaghan, p::AbstractVector) =
    PlaneCallaghan(a=p[1], D_intra=p[2], normal=p[3:5])

"""
    signal(plane::PlaneCallaghan, acq::Acquisition)

Compute the dMRI signal attenuation for restricted diffusion between
parallel planes using the GPD approximation.

For PGSE with pulse duration δ and diffusion time Δ:

    S/S₀ = Σₙ cₙ exp(-αₙ² D δ) [f(αₙ, δ, Δ)]

where αₙ = nπ/a are the eigenvalues of the 1D diffusion operator with
Neumann boundary conditions, and the sum runs over n = 1, 2, ...

The signal depends on the component of the gradient parallel to the plane
normal: q_par = g · n̂.
"""
function signal(plane::PlaneCallaghan, acq::Acquisition)
    (; a, D_intra, normal) = plane

    n_hat = normal ./ max(norm(normal), 1e-12)
    n_meas = length(acq.bvalues)
    S = ones(Float64, n_meas)

    delta = acq.delta === nothing ? 10e-3 : acq.delta
    Delta = acq.Delta === nothing ? 30e-3 : acq.Delta
    gamma = GYROMAGNETIC_RATIO

    N_TERMS = 20  # number of terms in the series

    for m in 1:n_meas
        b = acq.bvalues[m]
        b < 1e3 && continue  # skip b≈0

        # Gradient direction and magnitude
        g_dir = acq.gradient_directions[m, :]
        # Component of q parallel to plane normal
        G = sqrt(b / (gamma^2 * delta^2 * (Delta - delta / 3)))
        q_par = G * abs(dot(g_dir, n_hat))

        if q_par < 1e-12
            continue  # gradient perpendicular to normal → no restriction
        end

        # GPD series for parallel planes (Callaghan 1997)
        log_atten = 0.0
        for n in 1:N_TERMS
            alpha_n = n * π / a
            Dn = D_intra * alpha_n^2

            # GPD attenuation factor for PGSE
            # Cf. Balinov et al., J. Magn. Reson. A 104, 17-25 (1993)
            if Dn * delta > 1e-10
                term = (2 * gamma^2 * q_par^2 / (alpha_n^4 * a^2)) *
                       (2 * Dn * delta - 2 +
                        2 * exp(-Dn * delta) +
                        2 * exp(-Dn * Delta) -
                        exp(-Dn * (Delta - delta)) -
                        exp(-Dn * (Delta + delta))) / Dn^2
                log_atten -= term
            end
        end

        S[m] = exp(log_atten)
    end

    return S
end
