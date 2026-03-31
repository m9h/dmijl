"""
    Sequential (adaptive) experimental design.

Greedy acquisition: at each step, evaluate candidate measurements and
select the one that maximizes the expected information gain (via FIM
D-optimality or PCE) conditioned on the current posterior.

Reference: Foster et al. (2021) "Deep Adaptive Design", ICML, arXiv:2103.02438
"""

"""
    CandidateMeasurement

A single candidate dMRI measurement to evaluate for sequential design.
"""
struct CandidateMeasurement
    bvalue::Float64
    direction::Vector{Float64}
end

"""
    generate_candidates(design_space; n_candidates=50, seed=42) -> Vector{CandidateMeasurement}

Generate a set of candidate measurements spanning the design space.
"""
function generate_candidates(ds::DesignSpace; n_candidates::Int=50, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    dirs = electrostatic_directions(n_candidates; seed=seed)
    delta = ds.delta_range[1]
    Delta = ds.Delta_range[1]
    b_max = min(ds.b_range[2], max_bvalue(ds.G_max, delta, Delta))

    return [CandidateMeasurement(
                ds.b_range[1] + rand(rng) * (b_max - ds.b_range[1]),
                dirs[i, :]
            ) for i in 1:n_candidates]
end

"""
    sequential_design(model, design_space, prior_samples;
                      n_candidates=50, sigma=0.02, criterion=:D) -> Acquisition

Greedy sequential protocol design.

Starting from b=0 measurements, iteratively selects the next measurement
that maximizes D-optimality of the expected FIM conditioned on the current
set of prior samples.

# Arguments
- `model`: forward model (MultiCompartmentModel or ConstrainedModel)
- `design_space`: DesignSpace specification
- `prior_samples`: (n_params × N) prior parameter samples

# Returns
- `Acquisition`: the greedily designed protocol
"""
function sequential_design(model, design_space::DesignSpace,
                           prior_samples::AbstractMatrix;
                           n_candidates::Int=50,
                           sigma::Float64=0.02,
                           criterion::Symbol=:D,
                           verbose::Bool=false)
    ds = design_space
    delta = ds.delta_range[1]
    Delta = ds.Delta_range[1]
    n_total = ds.n_measurements

    # Start with b=0 measurements
    bvals = zeros(ds.n_b0)
    bvecs = repeat([1.0 0.0 0.0], ds.n_b0)

    candidates = generate_candidates(ds; n_candidates=n_candidates)

    for step in 1:(n_total - ds.n_b0)
        best_val = -Inf
        best_c = 1

        for (ic, c) in enumerate(candidates)
            # Augmented acquisition with this candidate
            trial_bvals = vcat(bvals, [c.bvalue])
            trial_dirs = vcat(bvecs, c.direction')
            trial_acq = Acquisition(trial_bvals, trial_dirs, delta, Delta)

            # Expected FIM over prior
            F = expected_fim(model, trial_acq, prior_samples;
                             sigma=sigma)
            val = optimality_criterion(F + 1e-12 * I; criterion=criterion)

            if val > best_val
                best_val = val
                best_c = ic
            end
        end

        # Add the best candidate
        c = candidates[best_c]
        push!(bvals, c.bvalue)
        bvecs = vcat(bvecs, c.direction')

        verbose && println("  step $step/$(n_total - ds.n_b0): b=$(round(c.bvalue, digits=0)), D-opt=$(round(best_val, digits=2))")
    end

    return Acquisition(bvals, bvecs, delta, Delta)
end
