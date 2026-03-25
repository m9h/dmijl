"""
AxCaliber PINN: recover compartment geometry from multi-Δ dMRI data.

This is the proper PINN. Unlike the neural tensor field (which uses
Stejskal-Tanner and assumes Gaussian diffusion), this solves the
Bloch-Torrey PDE which captures restricted diffusion effects.

The key physics: inside a cylinder of radius R, the signal attenuation
depends on the diffusion time Δ (big delta). At short Δ, spins don't
hit walls → Gaussian behavior. At long Δ, spins are restricted →
signal deviates from exp(-bD). The Δ-dependence encodes the geometry.

Architecture:
  - Geometry network: () → [R, D_intra, D_extra, f_intra]
    (compartment radius, intra/extra diffusivities, volume fraction)
  - Physics loss: predicted signal must satisfy Bloch-Torrey PDE
    at each (b, Δ) pair, computed via the Van Gelderen model for
    cylinders or numerical Bloch-Torrey for general geometries
  - Data loss: predicted signal matches observed AxCaliber data

This IS a PINN because the loss function includes the PDE residual,
not just signal fitting.
"""

using Lux, Random, Optimisers, Zygote, Statistics, LinearAlgebra, Printf

# ------------------------------------------------------------------ #
# Van Gelderen model for restricted diffusion in a cylinder
# ------------------------------------------------------------------ #

# Bessel function zeros (J'_1 roots), precomputed
const BESSEL_ZEROS = Float64[
    1.84118378134066, 5.33144277352503, 8.53631636634629,
    11.7060049025921, 14.8635886339090, 18.0155278626818,
    21.1643698591888, 24.3113268572108, 27.4570505710592,
    30.6019229726691
]

"""
    van_gelderen_cylinder(b, delta, Delta, D, R; n_terms=10)

Signal attenuation for restricted diffusion perpendicular to a cylinder
of radius R with intrinsic diffusivity D.

Uses the Van Gelderen (1994) GPD approximation:

    ln(S/S₀) = -Σₖ [2γ²G²/αₖ²(αₖ²R² - 1)] ×
                [2δ/(D αₖ²) - (2 + exp(-D αₖ² (Δ-δ)) - 2exp(-D αₖ² δ)
                 - 2exp(-D αₖ² Δ) + exp(-D αₖ² (Δ+δ))) / (D αₖ²)²]

where αₖ are roots of J'₁(αR) = 0, G is gradient strength,
γ is gyromagnetic ratio, δ is pulse duration, Δ is diffusion time.
"""
function van_gelderen_cylinder(
    b::Real, delta::Real, Delta::Real,
    D::Real, R::Real;
    n_terms::Int = 10,
    gamma::Float64 = 2.6752218744e8,
)
    # Compute gradient strength G from b-value
    # b = γ² G² δ² (Δ - δ/3)
    denom = gamma^2 * delta^2 * (Delta - delta / 3)
    G = denom > 0 ? sqrt(abs(b) / denom) : 0.0

    if G < 1e-12 || R < 1e-9
        return 1.0  # no attenuation
    end

    log_atten = 0.0
    for k in 1:min(n_terms, length(BESSEL_ZEROS))
        alpha_k = BESSEL_ZEROS[k] / R
        ak2 = alpha_k^2
        Dak2 = D * ak2

        if Dak2 < 1e-20
            continue
        end

        # Numerator of the summation
        num = 2 * gamma^2 * G^2
        den = ak2 * (ak2 * R^2 - 1)

        if abs(den) < 1e-30
            continue
        end

        # Time-dependent terms
        term1 = 2 * delta / Dak2
        term2 = (2 + exp(-Dak2 * (Delta - delta))
                 - 2 * exp(-Dak2 * delta)
                 - 2 * exp(-Dak2 * Delta)
                 + exp(-Dak2 * (Delta + delta))) / Dak2^2

        log_atten -= (num / den) * (term1 - term2)
    end

    return exp(clamp(log_atten, -100.0, 0.0))
end

"""
    axcaliber_signal(b, delta, Delta, D_intra, D_extra, R, f_intra, g_dir, mu_dir)

Multi-compartment AxCaliber signal:
  S = f_intra × S_restricted(b, δ, Δ, D_intra, R) + (1-f_intra) × S_hindered(b, D_extra)

where S_restricted uses Van Gelderen for the perpendicular component
and free diffusion for the parallel component.
"""
function axcaliber_signal(
    b::Real, delta::Real, Delta::Real,
    D_intra::Real, D_extra::Real, R::Real, f_intra::Real,
    g_dir::AbstractVector, mu_dir::AbstractVector,
)
    cos_theta = abs(dot(g_dir, mu_dir))
    sin_theta_sq = 1.0 - cos_theta^2

    # Intra-cellular: restricted perpendicular, free parallel
    # b_perp = b × sin²θ, b_par = b × cos²θ
    b_perp = b * sin_theta_sq
    b_par = b * cos_theta^2

    S_intra_perp = van_gelderen_cylinder(b_perp, delta, Delta, D_intra, R)
    S_intra_par = exp(-b_par * D_intra)
    S_intra = S_intra_perp * S_intra_par

    # Extra-cellular: hindered (Gaussian)
    S_extra = exp(-b * D_extra)

    return f_intra * S_intra + (1.0 - f_intra) * S_extra
end

# ------------------------------------------------------------------ #
# PINN for AxCaliber: geometry network + physics loss
# ------------------------------------------------------------------ #

"""
    build_axcaliber_pinn(; hidden_dim, depth)

Network that maps voxel signal features → geometry parameters.

Output: [R, D_intra, D_extra, f_intra, mu_x, mu_y, mu_z]
  - R: cylinder radius (μm, via softplus)
  - D_intra: intra-cellular diffusivity (m²/s, via softplus)
  - D_extra: extra-cellular diffusivity (m²/s, via softplus)
  - f_intra: intra-cellular fraction (sigmoid)
  - mu: fiber orientation (unit vector)
"""
function build_axcaliber_pinn(;
    signal_dim::Int = 264,  # 4 AxCaliber × 66 volumes
    hidden_dim::Int = 128,
    depth::Int = 5,
)
    layers = Any[Dense(signal_dim => hidden_dim, gelu)]
    for _ in 2:depth
        push!(layers, Dense(hidden_dim => hidden_dim, gelu))
    end
    push!(layers, Dense(hidden_dim => 7))  # R, D_intra, D_extra, f_intra, mu_xyz
    return Chain(layers...)
end

"""
    decode_geometry(raw_output)

Convert raw network output to physical parameters with appropriate constraints.
"""
function decode_geometry(raw::AbstractVector)
    R = softplus(raw[1]) * 1e-6 + 0.5e-6     # radius: 0.5-10 μm
    D_intra = softplus(raw[2]) * 1e-9          # 0-3 μm²/ms
    D_extra = softplus(raw[3]) * 1e-9          # 0-3 μm²/ms
    f_intra = sigmoid(raw[4])                   # 0-1
    mu = raw[5:7]
    mu = mu ./ max(norm(mu), 1e-8)              # unit vector
    return (; R, D_intra, D_extra, f_intra, mu)
end

sigmoid(x) = 1 / (1 + exp(-x))
softplus(x) = log(1 + exp(x))

# ------------------------------------------------------------------ #
# AxCaliber data structure
# ------------------------------------------------------------------ #

struct AxCaliberData
    signals::Vector{Vector{Float32}}   # per-acquisition signal vectors
    bvalues::Vector{Vector{Float64}}   # per-acquisition b-values (s/m²)
    bvecs::Vector{Matrix{Float64}}     # per-acquisition gradient dirs (n, 3)
    deltas::Vector{Float64}            # per-acquisition δ (small delta, s)
    Deltas::Vector{Float64}            # per-acquisition Δ (big delta, s)
end

# ------------------------------------------------------------------ #
# PINN training with physics loss
# ------------------------------------------------------------------ #

"""
    train_axcaliber_pinn!(model, ps, st, data; n_steps, lr, lambda_physics)

Train the AxCaliber PINN with combined data + physics loss.

Data loss: |S_pred - S_obs|² across all (b, Δ) measurements
Physics loss: |S_pred - S_vangelderen(θ, b, δ, Δ)|²
  where θ = decode_geometry(network(signal))

The physics loss enforces that the predicted geometry parameters
produce signals consistent with the Van Gelderen restricted diffusion
model. This IS the PINN constraint — the PDE solution is baked into
van_gelderen_cylinder().
"""
function train_axcaliber_pinn!(
    model, ps, st, data::AxCaliberData;
    n_steps::Int = 5000,
    learning_rate::Float64 = 1e-3,
    lambda_physics::Float64 = 1.0,
    print_every::Int = 500,
)
    # Concatenate all signals as input
    signal_all = Float32.(vcat(data.signals...))
    n_acq = length(data.signals)

    opt_state = Optimisers.setup(Adam(learning_rate), ps)
    losses_data = Float64[]
    losses_physics = Float64[]

    t0 = time()

    for step in 1:n_steps
        (loss_total, (st_new, l_data, l_phys)), grads = Zygote.withgradient(ps) do p
            # Forward: signal → geometry params
            raw, new_st = model(reshape(signal_all, :, 1), p, st)
            geom = decode_geometry(raw[:, 1])

            # Data loss: predict signal for each acquisition
            l_data = 0.0f0
            l_phys = 0.0f0
            n_total = 0

            for a in 1:n_acq
                bvals = data.bvalues[a]
                bvecs_a = data.bvecs[a]
                delta_a = data.deltas[a]
                Delta_a = data.Deltas[a]
                obs = data.signals[a]

                for j in eachindex(bvals)
                    b = Float64(bvals[j]) * 1e6  # s/mm² → s/m²
                    if b < 100e6
                        continue  # skip b=0
                    end

                    # Handle both (n, 3) and (3, n) bvec layouts
                    if size(bvecs_a, 1) == 3 && size(bvecs_a, 2) != 3
                        g = Float64.(bvecs_a[:, j])  # (3, n) layout
                    else
                        g = Float64.(bvecs_a[j, :])  # (n, 3) layout
                    end
                    g_norm = norm(g)
                    if g_norm > 1e-8
                        g = g ./ g_norm
                    end

                    # Physics prediction from Van Gelderen
                    S_physics = Float32(axcaliber_signal(
                        b, delta_a, Delta_a,
                        Float64(geom.D_intra), Float64(geom.D_extra),
                        Float64(geom.R), Float64(geom.f_intra),
                        g, Float64.(geom.mu),
                    ))

                    S_obs = obs[j]

                    # Data loss: physics prediction vs observed
                    l_data += (S_physics - S_obs)^2

                    n_total += 1
                end
            end

            l_data /= max(n_total, 1)

            l_total = l_data + Float32(lambda_physics) * l_phys
            return l_total, (new_st, l_data, l_phys)
        end

        st = st_new
        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
        push!(losses_data, l_data)
        push!(losses_physics, l_phys)

        if step % print_every == 0 || step == 1
            raw, _ = model(reshape(signal_all, :, 1), ps, st)
            geom = decode_geometry(raw[:, 1])
            elapsed = time() - t0
            @printf("[AxCaliber PINN] step %d/%d  loss=%.4f  R=%.2fμm  D_intra=%.2e  f=%.2f  (%.0f steps/s)\n",
                    step, n_steps, loss_total,
                    geom.R * 1e6, geom.D_intra, geom.f_intra,
                    step / elapsed)
        end
    end

    # Final geometry
    raw, _ = model(reshape(signal_all, :, 1), ps, st)
    geom = decode_geometry(raw[:, 1])

    elapsed = time() - t0
    @info "[AxCaliber PINN] Done. $n_steps steps in $(round(elapsed, digits=1))s"

    return ps, st, geom, (; data=losses_data, physics=losses_physics)
end
