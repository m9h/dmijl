"""
Out-of-distribution detection for SBI models.

Flags inputs that are far from the training distribution,
indicating the model's predictions may be unreliable.

Pure Julia — no neural networks or AD needed, just statistics
and linear algebra.
"""

using Statistics, LinearAlgebra

"""
    reconstruction_error(predict_fn, X) → Vector{Float64}

Per-sample MSE between `X` and `predict_fn(X)`.

`predict_fn(X)` should return a matrix of the same size as `X`
(rows = samples, cols = features).
"""
function reconstruction_error(predict_fn, X::AbstractMatrix)::Vector{Float64}
    X_hat = predict_fn(X)
    n = size(X, 1)
    d = size(X, 2)
    errs = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:d
            s += (X[i, j] - X_hat[i, j])^2
        end
        errs[i] = s / d
    end
    return errs
end

"""
    mahalanobis_distance(X, X_train) → Vector{Float64}

Compute the Mahalanobis distance of each row of `X` from the
distribution defined by `X_train` (fit mean + covariance from
`X_train`).
"""
function mahalanobis_distance(X::AbstractMatrix, X_train::AbstractMatrix)::Vector{Float64}
    mu = vec(mean(X_train, dims=1))         # (d,)
    C = cov(X_train, dims=1)                # (d, d)
    d = size(X_train, 2)
    # Regularise for numerical stability
    C_inv = inv(C + 1e-6 * I(d))

    n = size(X, 1)
    dists = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        diff = X[i, :] .- mu
        dists[i] = sqrt(max(dot(diff, C_inv * diff), 0.0))
    end
    return dists
end

"""
    ood_score(predict_fn, X, X_train; weights=(0.5, 0.5)) → Vector{Float64}

Weighted combination of reconstruction error and Mahalanobis distance.

`weights` is a tuple `(w_recon, w_mahal)`.
"""
function ood_score(
    predict_fn, X::AbstractMatrix, X_train::AbstractMatrix;
    weights::Tuple{Float64, Float64} = (0.5, 0.5),
)::Vector{Float64}
    w_r, w_m = weights
    re = reconstruction_error(predict_fn, X)
    md = mahalanobis_distance(X, X_train)
    return w_r .* re .+ w_m .* md
end

"""
    ood_detect(scores; threshold=nothing, percentile=95) → BitVector

Flag points whose score exceeds `threshold`.  If `threshold` is
`nothing`, it is automatically set from the given `percentile`
of `scores`.
"""
function ood_detect(
    scores::AbstractVector;
    threshold::Union{Nothing, Real} = nothing,
    percentile::Real = 95,
)::BitVector
    if threshold === nothing
        sorted = sort(collect(scores))
        idx = ceil(Int, percentile / 100 * length(sorted))
        idx = clamp(idx, 1, length(sorted))
        threshold = sorted[idx]
    end
    return scores .> threshold
end
