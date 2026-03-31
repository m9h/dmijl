"""
    Protocol optimization for dMRI experimental design.

Gradient-based optimization of b-values (and optionally timing parameters)
to maximize a chosen FIM-based optimality criterion.
"""

using Optim

"""
    optimize_protocol(problem::OEDProblem; n_restarts=5, max_iter=500) -> DesignResult

Optimize the dMRI acquisition protocol to maximize the chosen optimality criterion.

Optimizes continuous b-values within hardware constraints. Uses L-BFGS-B with
box constraints and multi-start initialization.
"""
function optimize_protocol(problem::OEDProblem;
                           n_restarts::Int=5,
                           max_iter::Int=500,
                           verbose::Bool=false)
    ds = problem.design_space
    n_dw = ds.n_measurements - ds.n_b0  # diffusion-weighted measurements

    # Generate fixed gradient directions
    dirs = electrostatic_directions(n_dw)
    b0_dirs = repeat([1.0 0.0 0.0], ds.n_b0)

    # Default timing
    delta = ds.delta_range[1]
    Delta = ds.Delta_range[1]
    b_max = min(ds.b_range[2], max_bvalue(ds.G_max, delta, Delta))

    best_result = nothing
    best_value = -Inf

    for restart in 1:n_restarts
        rng = Random.MersenneTwister(restart * 1337)

        # Random initial b-values in [b_min, b_max]
        x0 = ds.b_range[1] .+ rand(rng, n_dw) .* (b_max - ds.b_range[1])
        lower = fill(ds.b_range[1], n_dw)
        upper = fill(b_max, n_dw)

        function objective(x)
            bvals = vcat(zeros(ds.n_b0), x)
            all_dirs = vcat(b0_dirs, dirs)
            acq = Acquisition(bvals, all_dirs, delta, Delta)

            if problem.prior_samples !== nothing
                F = expected_fim(problem.model, acq, problem.prior_samples;
                                 sigma=problem.sigma, noise_model=problem.noise_model)
            else
                theta_nom = _nominal_params(problem.model)
                F = fisher_information(problem.model, acq, theta_nom;
                                       sigma=problem.sigma, noise_model=problem.noise_model)
            end

            F_reg = F + 1e-12 * I
            return -optimality_criterion(F_reg; criterion=problem.criterion,
                                          weights=problem.weights)
        end

        result = Optim.optimize(objective, lower, upper, x0,
                                Optim.Fminbox(Optim.LBFGS()),
                                Optim.Options(iterations=max_iter, g_tol=1e-8))

        val = -Optim.minimum(result)
        if val > best_value
            best_value = val
            x_opt = Optim.minimizer(result)
            bvals_opt = vcat(zeros(ds.n_b0), x_opt)
            all_dirs = vcat(b0_dirs, dirs)
            acq_opt = Acquisition(bvals_opt, all_dirs, delta, Delta)

            theta_nom = problem.prior_samples !== nothing ?
                vec(mean(problem.prior_samples, dims=2)) :
                _nominal_params(problem.model)
            F_opt = fisher_information(problem.model, acq_opt, theta_nom;
                                       sigma=problem.sigma, noise_model=problem.noise_model)

            best_result = DesignResult(
                acq_opt,
                F_opt,
                diag(inv(F_opt + 1e-12 * I)),
                best_value,
                collect(String.(parameter_names(problem.model)))
            )
        end

        verbose && println("  restart $restart: criterion = $(round(val, digits=3))")
    end

    return best_result
end

"""
    _nominal_params(model) -> Vector{Float64}

Midpoint of parameter ranges as a nominal parameter vector for FIM evaluation.
"""
function _nominal_params(model)
    lo, hi = get_flat_bounds(model)
    return (lo .+ hi) ./ 2
end
