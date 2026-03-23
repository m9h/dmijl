"""
Evaluation metrics for microstructure parameter estimation.
"""

using Statistics, LinearAlgebra

"""Angular error between two unit vectors (handles antipodal symmetry)."""
function angular_error_deg(mu_true::AbstractVector, mu_pred::AbstractVector)
    dot_val = abs(dot(mu_true, mu_pred))
    return rad2deg(acos(clamp(dot_val, 0.0, 1.0)))
end

"""Pearson correlation coefficient."""
function pearson_r(x::AbstractVector, y::AbstractVector)
    n = length(x)
    mx, my = mean(x), mean(y)
    sx = sqrt(sum((x .- mx).^2) / n)
    sy = sqrt(sum((y .- my).^2) / n)
    (sx == 0 || sy == 0) && return 0.0
    return mean((x .- mx) .* (y .- my)) / (sx * sy)
end

"""Root mean squared error."""
rmse(pred, true_val) = sqrt(mean((pred .- true_val).^2))

"""
    evaluate_ball2stick(theta_true, theta_pred; n_scalars=4)

Compute standard Ball+2Stick evaluation metrics:
correlations for scalar params, orientation errors for fibers.
Handles label-switching symmetry.
"""
function evaluate_ball2stick(
    theta_true::AbstractMatrix,   # (n_params, n_test)
    theta_pred::AbstractMatrix;
    n_scalars::Int=4,
)
    n_test = size(theta_true, 2)
    scalar_names = ["d_ball", "d_stick", "f1", "f2"]

    # Scalar correlations
    correlations = Dict{String, Float64}()
    for (i, name) in enumerate(scalar_names)
        correlations[name] = pearson_r(theta_true[i, :], theta_pred[i, :])
    end

    # Orientation errors with label-switching
    errors1 = zeros(n_test)
    errors2 = zeros(n_test)

    for j in 1:n_test
        mu1_t = theta_true[5:7, j]
        mu1_p = theta_pred[5:7, j]
        mu1_p = mu1_p ./ max(norm(mu1_p), 1e-8)
        mu2_t = theta_true[8:10, j]
        mu2_p = theta_pred[8:10, j]
        mu2_p = mu2_p ./ max(norm(mu2_p), 1e-8)

        e11 = angular_error_deg(mu1_t, mu1_p)
        e22 = angular_error_deg(mu2_t, mu2_p)
        e12 = angular_error_deg(mu1_t, mu2_p)
        e21 = angular_error_deg(mu2_t, mu1_p)

        if e11 + e22 <= e12 + e21
            errors1[j] = e11
            errors2[j] = e22
        else
            errors1[j] = e12
            errors2[j] = e21
        end
    end

    return (;
        correlations,
        fiber1_median = median(errors1),
        fiber1_mean = mean(errors1),
        fiber2_median = median(errors2),
        fiber2_mean = mean(errors2),
        errors1, errors2,
    )
end
