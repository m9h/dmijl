"""
Non-parametric diffusion field recovery v2 — direction-aware.

Key fix: the signal prediction is now direction-dependent.

For each measurement (b, g), the predicted signal is:
    S(b, g) = ∫_voxel exp(-b · gᵀ D(x) g) dx

where D(x) is the learned diffusion tensor field and the integral
is approximated by Monte Carlo sampling over spatial positions.

This connects gradient direction to D(x) anisotropy, so FA > 0
when the data supports it.
"""

using Lux, Random, Optimisers, Zygote, Statistics, LinearAlgebra

# Types from v1 (diffusion_field.jl) are already loaded by the module

# ------------------------------------------------------------------ #
# Direction-aware signal prediction
# ------------------------------------------------------------------ #

"""
    predict_signal_directional(D_net, ps_D, st_D, D_type, b, g, x_samples)

Predict dMRI signal for a single measurement (b-value `b`, gradient
direction `g`) by integrating the Gaussian diffusion kernel over
spatial positions in the voxel:

    S(b, g) ≈ (1/N) Σᵢ exp(-b · gᵀ D(xᵢ) g)

This is the Stejskal-Tanner equation applied voxel-wise with a
spatially-varying D(x) from the D-network.
"""
function predict_signal_directional(
    D_net, ps_D, st_D, D_type::Symbol,
    b::Float32, g::AbstractVector{Float32},
    x_samples::AbstractMatrix{Float32},  # (3, n_spatial)
)
    n = size(x_samples, 2)
    total = 0.0f0

    for i in 1:n
        xi = @view x_samples[:, i]
        D = eval_D(D_net, ps_D, st_D, xi, D_type)

        if D_type == :scalar
            # Isotropic: S = exp(-b D)
            adc = D
        elseif D_type == :diagonal
            # Diagonal tensor: ADC = gᵀ diag(D) g = Σ gₖ² Dₖ
            adc = g[1]^2 * D[1] + g[2]^2 * D[2] + g[3]^2 * D[3]
        else  # :full
            # Full tensor: ADC = gᵀ D g
            adc = dot(g, D * g)
        end

        total += exp(-b * adc)
    end

    return total / n
end

# ------------------------------------------------------------------ #
# Solver v2: direction-aware
# ------------------------------------------------------------------ #

"""
    solve_diffusion_field_v2(problem; ...)

Recover D(x) using direction-aware signal prediction.

Each measurement's predicted signal depends on its specific
(b-value, gradient direction) pair, enabling FA recovery.
"""
function solve_diffusion_field_v2(
    problem::DiffusionFieldProblem;
    output_type::Symbol = :diagonal,
    D_hidden::Int = 64,
    D_depth::Int = 4,
    n_steps::Int = 10_000,
    n_spatial::Int = 32,
    n_meas_per_step::Int = 20,
    learning_rate::Float64 = 1e-3,
    print_every::Int = 1000,
    seed::Int = 42,
)
    rng = MersenneTwister(seed)

    D_net, D_type = build_diffusivity_net(;
        hidden_dim=D_hidden, depth=D_depth, output_type=output_type)
    ps, st = Lux.setup(rng, D_net)
    opt_state = Optimisers.setup(Adam(learning_rate), ps)

    vr = Float32(problem.voxel_size / 2)
    n_meas = length(problem.observed_signal)

    # Precompute gradient directions as Float32
    # bvecs may be (3, n_meas) or (n_meas, 3) — normalize
    gdir = problem.gradient_directions
    if size(gdir, 1) == 3 && size(gdir, 2) == n_meas
        gdir = gdir'  # → (n_meas, 3)
    end
    # Normalize each direction
    for i in 1:n_meas
        n = norm(@view gdir[i, :])
        if n > 1e-8
            gdir[i, :] ./= n
        end
    end
    gdir_f32 = Float32.(gdir)  # (n_meas, 3)
    bvals_f32 = Float32.(problem.bvalues)

    losses = Float64[]
    t0 = time()

    for step in 1:n_steps
        # Random spatial positions in voxel
        x_samples = (2.0f0 .* rand(rng, Float32, 3, n_spatial) .- 1.0f0) .* vr

        # Subsample measurements (cycle through all over training)
        meas_indices = [((step - 1) * n_meas_per_step + m - 1) % n_meas + 1
                        for m in 1:n_meas_per_step]

        (loss, st), grads = Zygote.withgradient(ps) do p
            l = 0.0f0
            current_st = st

            for idx in meas_indices
                b = bvals_f32[idx]
                g = gdir_f32[idx, :]
                S_obs = problem.observed_signal[idx]

                # Skip b=0 (no diffusion weighting → always 1)
                if b < 100f0
                    S_pred = 1.0f0
                else
                    S_pred = predict_signal_directional(
                        D_net, p, current_st, D_type,
                        b, g, x_samples,
                    )
                end

                l += (S_pred - S_obs)^2
            end

            return l / n_meas_per_step, current_st
        end

        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
        push!(losses, loss)

        if step % print_every == 0 || step == 1
            elapsed = time() - t0
            rate = step / elapsed

            # Quick D evaluation at center
            D_c = eval_D(D_net, ps, st, Float32[0,0,0], D_type)
            if D_type == :diagonal
                ds = sort(D_c, rev=true)
                md = mean(ds)
                fa = md > 0 ? sqrt(3/2) * sqrt(sum((ds .- md).^2)) / sqrt(sum(ds.^2)) : 0.0
                @info "[D-field v2] step $step/$n_steps  loss=$(round(loss, sigdigits=3))  " *
                      "MD=$(round(md, sigdigits=3))  FA=$(round(fa, digits=3))  " *
                      "($(round(rate, digits=0)) steps/s)"
            else
                @info "[D-field v2] step $step/$n_steps  loss=$(round(loss, sigdigits=3))  " *
                      "($(round(rate, digits=0)) steps/s)"
            end
        end
    end

    elapsed = time() - t0
    @info "[D-field v2] Done. $n_steps steps in $(round(elapsed, digits=1))s"

    return (; D_net, ps_D=ps, st_D=st, D_type=output_type, losses)
end
