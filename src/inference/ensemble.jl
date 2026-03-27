"""
Deep Ensemble wrapper for any inference method.

Trains N independent models with different random seeds and aggregates their
predictions. Provides calibrated uncertainty via inter-model disagreement.

Reference: Lakshminarayanan et al. (2017) "Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles".
"""

# ─── Training ─────────────────────────────────────────────────────────────────

"""
    train_ensemble(build_fn, train_fn, n_members; seeds=nothing)

Train an ensemble of `n_members` independent models.

# Arguments
- `build_fn`: `() -> (model, ps, st)` — creates a fresh model with random init
- `train_fn`: `(model, ps, st, seed) -> (ps, st, losses)` — trains one member
- `n_members`: number of ensemble members
- `seeds`: optional vector of integer seeds (length `n_members`); defaults to `1:n_members`

# Returns
- `Vector` of `(model, ps, st)` tuples, one per trained member
"""
function train_ensemble(build_fn, train_fn, n_members::Int; seeds=nothing)
    seeds = isnothing(seeds) ? collect(1:n_members) : seeds
    @assert length(seeds) == n_members "seeds must have length n_members"

    ensemble = Vector{Tuple}(undef, n_members)
    for i in 1:n_members
        model, ps, st = build_fn()
        ps, st, _ = train_fn(model, ps, st, seeds[i])
        ensemble[i] = (model, ps, st)
    end
    return ensemble
end

# ─── Prediction ───────────────────────────────────────────────────────────────

"""
    ensemble_predict(ensemble, predict_fn, x)

Run `predict_fn` on each ensemble member and collect the results.

# Arguments
- `ensemble`: vector of `(model, ps, st)` tuples
- `predict_fn`: `(model, ps, st, x) -> prediction`
- `x`: input data

# Returns
- `Vector` of predictions, one per member
"""
function ensemble_predict(ensemble, predict_fn, x)
    return [predict_fn(m, p, s, x) for (m, p, s) in ensemble]
end

# ─── Aggregation ──────────────────────────────────────────────────────────────

"""
    ensemble_mean(predictions)

Element-wise mean across ensemble member predictions.
"""
function ensemble_mean(predictions)
    return mean(predictions)
end

"""
    ensemble_std(predictions)

Element-wise standard deviation across ensemble member predictions.
Captures inter-model disagreement (epistemic uncertainty).
"""
function ensemble_std(predictions)
    mu = ensemble_mean(predictions)
    n = length(predictions)
    # Population std (matches std with corrected=false)
    sq_diffs = [(p .- mu).^2 for p in predictions]
    return sqrt.(sum(sq_diffs) ./ n)
end

# ─── Sampling ─────────────────────────────────────────────────────────────────

"""
    ensemble_sample(ensemble, sample_fn, x, rng; n_samples_per_member=100)

Draw samples from each ensemble member and concatenate.

# Arguments
- `ensemble`: vector of `(model, ps, st)` tuples
- `sample_fn`: `(model, ps, st, x, rng; n_samples) -> matrix (D, n_samples)`
- `x`: input data
- `rng`: random number generator
- `n_samples_per_member`: samples to draw from each member

# Returns
- Matrix of shape `(D, n_members * n_samples_per_member)`
"""
function ensemble_sample(ensemble, sample_fn, x, rng::AbstractRNG;
    n_samples_per_member::Int=100,
)
    parts = [sample_fn(m, p, s, x, rng; n_samples=n_samples_per_member)
             for (m, p, s) in ensemble]
    return hcat(parts...)
end
