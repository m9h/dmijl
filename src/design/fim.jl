"""
    Fisher Information Matrix computation for dMRI forward models.

Uses ForwardDiff.jl for exact Jacobians through the `signal()` dispatch.
Supports Gaussian and Rician noise models.
"""

using ForwardDiff
using SpecialFunctions: besseli

"""
    jacobian_signal(model, acq, params) -> Matrix{Float64}

Compute J[m, i] = ∂S_m/∂θ_i via finite differences.
Returns (n_measurements × n_params) matrix.

Uses central finite differences for robustness with struct-typed
compartment models that don't support ForwardDiff Dual numbers.
"""
function jacobian_signal(model, acq::Acquisition, params::AbstractVector)
    n_meas = length(acq.bvalues)
    n_params = length(params)
    J = zeros(n_meas, n_params)
    eps = 1e-7

    for j in 1:n_params
        p_plus = copy(params); p_plus[j] += eps
        p_minus = copy(params); p_minus[j] -= eps
        J[:, j] .= (signal(model, acq, p_plus) .- signal(model, acq, p_minus)) ./ (2 * eps)
    end
    return J
end

"""
    rician_fim_correction(S, sigma) -> Float64

Rician FIM correction factor L(S,σ).
At high SNR → 1 (Gaussian limit). At low SNR → 0 (no information).
Ref: Alexander 2008, Eq. 7.
"""
function rician_fim_correction(S::Real, sigma::Real)
    S < 1e-10 && return 0.0
    z = S^2 / (2 * sigma^2)
    z > 500.0 && return 1.0
    ratio = besseli(1, z) / besseli(0, z)
    return 1.0 - (2 * sigma^2 / S^2) * (1.0 - ratio)
end

"""
    fisher_information(model, acq, params; sigma=0.02, noise_model=:gaussian)

Compute the Fisher Information Matrix.

- Gaussian: F = (1/σ²) J'J
- Rician: F = J' diag(L/σ²) J where L is the Rician correction per measurement
"""
function fisher_information(model, acq::Acquisition, params::AbstractVector;
                            sigma::Float64=0.02, noise_model::Symbol=:gaussian)
    J = jacobian_signal(model, acq, params)
    if noise_model == :gaussian
        return (1.0 / sigma^2) * J' * J
    elseif noise_model == :rician
        S = signal(model, acq, params)
        L = rician_fim_correction.(S, sigma)
        W = Diagonal(L ./ sigma^2)
        return J' * W * J
    else
        error("Unknown noise_model: $noise_model. Use :gaussian or :rician.")
    end
end

"""
    expected_fim(model, acq, prior_samples; sigma=0.02, noise_model=:gaussian)

Expected FIM averaged over prior samples (Bayesian version).
`prior_samples` is (n_params × n_samples).
"""
function expected_fim(model, acq::Acquisition, prior_samples::AbstractMatrix;
                      sigma::Float64=0.02, noise_model::Symbol=:gaussian)
    n_params, n_samples = size(prior_samples)
    F_avg = zeros(n_params, n_params)
    for i in 1:n_samples
        F_avg .+= fisher_information(model, acq, prior_samples[:, i];
                                     sigma=sigma, noise_model=noise_model)
    end
    return F_avg ./ n_samples
end

"""
    crlb(model, acq, params; sigma=0.02, noise_model=:gaussian) -> Vector{Float64}

Cramér-Rao Lower Bound: diag(F⁻¹). Minimum achievable variance per parameter.
"""
function crlb(model, acq::Acquisition, params::AbstractVector;
              sigma::Float64=0.02, noise_model::Symbol=:gaussian)
    F = fisher_information(model, acq, params; sigma=sigma, noise_model=noise_model)
    return diag(inv(F + 1e-12 * I))
end
