"""
    Conformal prediction intervals for parameter estimates.

Split conformal inference provides distribution-free, finite-sample valid
coverage guarantees that work regardless of model calibration.

Two methods:
- `split_conformal`: symmetric intervals around point predictions (absolute residuals).
- `cqr_conformal`: Conformalized Quantile Regression with adaptive intervals.

References
----------
Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a
Random World. Springer.

Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized Quantile
Regression. NeurIPS.

Angelopoulos, A. N. & Bates, S. (2023). Conformal Prediction: A Gentle
Introduction. Foundations and Trends in Machine Learning.
"""

using Statistics

"""
    split_conformal(predict_fn, X_cal, Y_cal, X_test; alpha=0.1) -> (lower, upper)

Split conformal prediction with absolute residual nonconformity scores.

# Arguments
- `predict_fn`: function `(X) -> Y_hat` returning point predictions, shape `(n, d_out)`.
- `X_cal`: calibration inputs, shape `(n_cal, d_in)`.
- `Y_cal`: calibration targets, shape `(n_cal, d_out)`.
- `X_test`: test inputs, shape `(n_test, d_in)`.
- `alpha`: miscoverage rate (default 0.1 for 90% coverage).

# Returns
- `(lower, upper)`: each of shape `(n_test, d_out)`.

The finite-sample coverage guarantee is:
    P(Y_test in [lower, upper]) >= 1 - alpha
"""
function split_conformal(predict_fn, X_cal::AbstractMatrix, Y_cal::AbstractMatrix,
                         X_test::AbstractMatrix; alpha::Real=0.1)
    n_cal = size(Y_cal, 1)
    d_out = size(Y_cal, 2)

    # Compute calibration predictions and absolute residuals
    Y_hat_cal = predict_fn(X_cal)
    residuals = abs.(Y_cal .- Y_hat_cal)  # (n_cal, d_out)

    # Conformal quantile level: ceil((1-alpha)(n_cal+1)) / n_cal
    # Equivalent to the (1-alpha)(1+1/n_cal) quantile, clipped to [0, 1]
    quantile_level = min((1 - alpha) * (1 + 1 / n_cal), 1.0)

    # Per-output-dimension quantile of residuals
    q_hat = Vector{Float64}(undef, d_out)
    for j in 1:d_out
        q_hat[j] = quantile(residuals[:, j], quantile_level)
    end

    # Compute test predictions and form intervals
    Y_hat_test = predict_fn(X_test)
    lower = Y_hat_test .- q_hat'
    upper = Y_hat_test .+ q_hat'

    return (lower, upper)
end

"""
    cqr_conformal(predict_lower_fn, predict_upper_fn, X_cal, Y_cal, X_test; alpha=0.1) -> (lower, upper)

Conformalized Quantile Regression (CQR).

Instead of symmetric intervals around a point prediction, CQR uses predicted
lower and upper quantile bounds and adjusts them with a conformal correction.

# Arguments
- `predict_lower_fn`: function `(X) -> Y_lower`, predicted lower quantile bounds.
- `predict_upper_fn`: function `(X) -> Y_upper`, predicted upper quantile bounds.
- `X_cal`: calibration inputs, shape `(n_cal, d_in)`.
- `Y_cal`: calibration targets, shape `(n_cal, d_out)`.
- `X_test`: test inputs, shape `(n_test, d_in)`.
- `alpha`: miscoverage rate (default 0.1).

# Returns
- `(lower, upper)`: each of shape `(n_test, d_out)`.

The nonconformity score is `max(predict_lower - Y, Y - predict_upper)`, which
measures how far Y falls outside the predicted interval (negative = inside).
"""
function cqr_conformal(predict_lower_fn, predict_upper_fn,
                       X_cal::AbstractMatrix, Y_cal::AbstractMatrix,
                       X_test::AbstractMatrix; alpha::Real=0.1)
    n_cal = size(Y_cal, 1)
    d_out = size(Y_cal, 2)

    # Compute calibration quantile predictions
    lower_cal = predict_lower_fn(X_cal)  # (n_cal, d_out)
    upper_cal = predict_upper_fn(X_cal)  # (n_cal, d_out)

    # CQR nonconformity scores: max(lower - Y, Y - upper)
    scores = max.(lower_cal .- Y_cal, Y_cal .- upper_cal)  # (n_cal, d_out)

    # Conformal quantile level
    quantile_level = min((1 - alpha) * (1 + 1 / n_cal), 1.0)

    # Per-output-dimension quantile of scores
    q_hat = Vector{Float64}(undef, d_out)
    for j in 1:d_out
        q_hat[j] = quantile(scores[:, j], quantile_level)
    end

    # Compute test quantile predictions and adjust by conformal correction
    lower_test = predict_lower_fn(X_test) .- q_hat'
    upper_test = predict_upper_fn(X_test) .+ q_hat'

    return (lower_test, upper_test)
end

"""
    conformal_coverage(lower, upper, Y_true) -> Float64

Compute the empirical coverage: fraction of samples where `lower <= Y_true <= upper`
across all output dimensions simultaneously (joint coverage per sample).

For multi-output, a sample is "covered" only if ALL dimensions are within their
respective intervals.
"""
function conformal_coverage(lower::AbstractMatrix, upper::AbstractMatrix,
                            Y_true::AbstractMatrix)
    # Per-element coverage
    covered = (Y_true .>= lower) .& (Y_true .<= upper)
    # Joint coverage: all dimensions covered per sample
    all_covered = vec(all(covered; dims=2))
    return mean(all_covered)
end
