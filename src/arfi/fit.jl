"""
    Closed-loop optimization for MR-ARFI.

Optimizes MSG sequence parameters and estimates tissue shear modulus
from observed displacement fields, enabling joint optimization of
ultrasound and MRI parameters.

Uses Optim.jl following the pattern in fem/fit.jl.
"""

using Optim

"""
    optimize_msg_params(intensity, labels, grid_spacing;
                        target_idx=nothing, max_iter=50) -> NamedTuple

Optimize MSG gradient parameters (amplitude, duration) to maximize
displacement sensitivity at the focal target.

Balances encoding strength (γGδ) against practical constraints:
- Scanner gradient limits (max 80 mT/m)
- Minimum TE increases with MSG duration (T2* decay)
- Phase noise floor (~0.05 rad)

# Arguments
- `intensity`: acoustic intensity field (W/m^2)
- `labels`: tissue labels
- `grid_spacing`: voxel size (metres)
- `target_idx`: CartesianIndex of focal target (default: argmax of intensity)
- `max_iter`: maximum optimization iterations

# Returns
Named tuple with optimized ARFISequenceParams, displacement, and loss history.
"""
function optimize_msg_params(
    intensity::AbstractArray{<:Real},
    labels::AbstractArray{<:Integer},
    grid_spacing::Float64;
    target_idx::Union{Nothing, CartesianIndex} = nothing,
    max_iter::Int = 50,
    verbose::Bool = true,
)
    if target_idx === nothing
        target_idx = argmax(Float64.(intensity))
    end

    # Fixed tissue properties
    c, _, alpha_db = map_labels_to_acoustic(labels)
    c = reshape(c, size(labels))
    alpha_db = reshape(alpha_db, size(labels))
    alpha_np = db_cm_to_neper_m.(alpha_db)
    mu = map_labels_to_shear_modulus(labels)
    mu = reshape(mu, size(labels))

    # Pre-compute displacement (independent of MSG params)
    force = compute_radiation_force(Float64.(intensity), c, alpha_np)
    displacement = solve_displacement_spectral(force, mu, grid_spacing)
    u_target = displacement[target_idx]

    # Optimize: maximize SNR = |γGδu| / σ_phase
    # where σ_phase ∝ exp(TE/T2*) and TE ∝ 2*msg_duration + fus_duration
    # Simplified: maximize encoding_coeff * |u_target|, penalize long TE
    T2star = 0.03  # 30 ms (typical brain)

    # x = [msg_amplitude (T/m), msg_duration (s)]
    x0 = [40e-3, 5e-3]
    lower = [5e-3, 1e-3]     # 5 mT/m, 1 ms
    upper = [80e-3, 20e-3]   # 80 mT/m, 20 ms

    loss_history = Float64[]

    function objective(x)
        G_msg, delta_msg = x
        enc = GAMMA_PROTON * G_msg * delta_msg
        phase_signal = enc * abs(u_target)

        # TE penalty: longer MSG → longer TE → more T2* decay
        te_eff = 2 * delta_msg + 10e-3  # 2*MSG + FUS window
        t2star_penalty = exp(-te_eff / T2star)

        # SNR ∝ phase_signal * t2star_penalty / phase_noise
        snr = phase_signal * t2star_penalty / 0.05

        loss = -snr  # minimize negative SNR
        push!(loss_history, loss)

        if verbose
            @printf("  G=%.1f mT/m, δ=%.1f ms, enc=%.0f rad/m, SNR=%.2f\n",
                    G_msg * 1e3, delta_msg * 1e3, enc, snr)
        end
        return loss
    end

    result = Optim.optimize(objective, lower, upper, x0,
                            Optim.Fminbox(Optim.NelderMead()),
                            Optim.Options(iterations = max_iter))

    G_opt, delta_opt = Optim.minimizer(result)

    opt_params = ARFISequenceParams(
        msg_amplitude = G_opt,
        msg_duration = delta_opt,
    )

    return (
        seq_params = opt_params,
        msg_amplitude = G_opt,
        msg_duration = delta_opt,
        encoding_coefficient = GAMMA_PROTON * G_opt * delta_opt,
        displacement_at_target = u_target,
        phase_at_target = GAMMA_PROTON * G_opt * delta_opt * u_target,
        loss_history = loss_history,
        converged = Optim.converged(result),
    )
end

"""
    fit_shear_modulus(observed_displacement, force_field, grid_spacing;
                     initial_mu=1.0e3, max_iter=100) -> NamedTuple

Estimate tissue shear modulus from observed displacement and known force.

Inverts the relationship μ∇²u = -F by finding μ that minimizes the
residual between observed and predicted displacement fields.

# Arguments
- `observed_displacement`: measured displacement from MR-ARFI (metres)
- `force_field`: known radiation force (N/m^3)
- `grid_spacing`: voxel size (metres)
- `initial_mu`: initial guess for shear modulus (Pa)

# Returns
Named tuple with fitted shear modulus and residual.
"""
function fit_shear_modulus(
    observed_displacement::AbstractArray{<:Real},
    force_field::AbstractArray{<:Real},
    grid_spacing::Float64;
    initial_mu::Float64 = 1.0e3,
    max_iter::Int = 100,
    verbose::Bool = true,
)
    function objective(log_mu_vec)
        mu = exp(log_mu_vec[1])  # optimize in log space for positivity
        u_pred = solve_displacement_spectral(force_field, mu, grid_spacing)
        return sum((u_pred .- observed_displacement) .^ 2)
    end

    x0 = [log(initial_mu)]
    result = Optim.optimize(objective, x0,
                            Optim.NelderMead(),
                            Optim.Options(iterations = max_iter))

    mu_fit = exp(Optim.minimizer(result)[1])
    u_fit = solve_displacement_spectral(force_field, mu_fit, grid_spacing)
    residual = sum((u_fit .- observed_displacement) .^ 2) / length(observed_displacement)

    if verbose
        println("Shear modulus fit: μ = $(round(mu_fit, sigdigits=4)) Pa " *
                "($(round(mu_fit/1e3, sigdigits=3)) kPa), MSE = $(round(residual, sigdigits=4))")
    end

    return (
        shear_modulus = mu_fit,
        displacement_fit = u_fit,
        residual = residual,
        converged = Optim.converged(result),
    )
end
