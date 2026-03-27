"""
Tests for Deep Ensemble wrapper around MDN (or any inference method).

Follows Red-Green TDD: tests written first, implementation to follow.
Tests: train_ensemble, ensemble_predict, ensemble_mean, ensemble_std,
       ensemble_sample, OOD disagreement, and integration with MDN.
"""

using Test, Random, Statistics, LinearAlgebra
using Lux, Zygote, Optimisers, ComponentArrays

# Include MDN source (standalone, same pattern as test_mdn.jl)
include("../src/inference/mdn.jl")
include("../src/inference/ensemble.jl")

@testset "Deep Ensembles" begin

    # ---- Shared dimensions for fast tests ----
    obs_dim = 4
    param_dim = 2
    n_components = 2
    hidden_dim = 16
    depth = 2

    @testset "train_ensemble returns vector of N trained parameter sets" begin
        n_members = 3

        # Simple synthetic data
        rng_data = MersenneTwister(1)
        N = 64
        true_params = randn(rng_data, Float32, param_dim, N)
        W = randn(rng_data, Float32, obs_dim, param_dim)
        signals = W * true_params

        build_fn = () -> begin
            model = build_mdn(; obs_dim=obs_dim, param_dim=param_dim,
                n_components=n_components, hidden_dim=hidden_dim, depth=depth)
            rng_init = MersenneTwister(rand(UInt32))
            ps, st = Lux.setup(rng_init, model)
            return (model, ps, st)
        end

        train_fn = (model, ps, st, seed) -> begin
            rng_t = MersenneTwister(seed)
            ps, st, losses = train_mdn!(model, ps, st, true_params, signals, rng_t;
                n_epochs=5, batch_size=32, lr=1e-3,
                n_components=n_components, param_dim=param_dim)
            return (ps, st, losses)
        end

        ensemble = train_ensemble(build_fn, train_fn, n_members; seeds=[10, 20, 30])

        @test length(ensemble) == n_members
        # Each element is a (model, ps, st) tuple
        for (m, p, s) in ensemble
            @test m isa Lux.AbstractLuxLayer
            @test p !== nothing
            @test s !== nothing
        end
        # Members should have different parameters (different seeds -> different init + training)
        ps1 = ensemble[1][2]
        ps2 = ensemble[2][2]
        # ComponentArrays can be compared element-wise via getdata or similar;
        # just check they are not identical objects
        @test ps1 !== ps2
    end

    @testset "ensemble_predict produces predictions from all members" begin
        n_members = 2

        # Build a tiny ensemble manually for this unit test
        models = []
        for i in 1:n_members
            model = build_mdn(; obs_dim=obs_dim, param_dim=param_dim,
                n_components=n_components, hidden_dim=hidden_dim, depth=depth)
            ps, st = Lux.setup(MersenneTwister(i), model)
            push!(models, (model, ps, st))
        end

        x = randn(MersenneTwister(99), Float32, obs_dim, 1)

        predict_fn = (model, ps, st, x) -> begin
            pi_w, mu, _, _ = mdn_forward(model, ps, st, x;
                n_components=n_components, param_dim=param_dim)
            # Mixture mean: sum_k pi_k * mu_k -> (param_dim,)
            return dropdims(sum(pi_w' .* mu[:, :, 1], dims=2), dims=2)
        end

        preds = ensemble_predict(models, predict_fn, x)

        @test length(preds) == n_members
        for p in preds
            @test length(p) == param_dim
            @test all(isfinite, p)
        end
    end

    @testset "ensemble_mean averages predictions across members" begin
        # Simple numeric test
        preds = [Float32[1.0, 2.0], Float32[3.0, 4.0], Float32[5.0, 6.0]]
        m = ensemble_mean(preds)
        @test m ≈ Float32[3.0, 4.0]
    end

    @testset "ensemble_std captures inter-model disagreement" begin
        # When members agree, std should be low
        preds_agree = [Float32[1.0, 2.0], Float32[1.0, 2.0], Float32[1.0, 2.0]]
        s_agree = ensemble_std(preds_agree)
        @test all(s_agree .< 1e-6)

        # When members disagree, std should be higher
        preds_disagree = [Float32[0.0, 0.0], Float32[5.0, 5.0], Float32[10.0, 10.0]]
        s_disagree = ensemble_std(preds_disagree)
        @test all(s_disagree .> 1.0)
    end

    @testset "ensemble_sample draws from all members equally" begin
        n_members = 3
        n_samples_per = 10

        # Build tiny ensemble
        ensemble = []
        for i in 1:n_members
            model = build_mdn(; obs_dim=obs_dim, param_dim=param_dim,
                n_components=n_components, hidden_dim=hidden_dim, depth=depth)
            ps, st = Lux.setup(MersenneTwister(i), model)
            push!(ensemble, (model, ps, st))
        end

        x = randn(MersenneTwister(42), Float32, obs_dim, 1)

        sample_fn = (model, ps, st, x, rng; n_samples) -> begin
            sample_mdn(model, ps, st, x, rng;
                n_samples=n_samples, n_components=n_components, param_dim=param_dim)
        end

        samples = ensemble_sample(ensemble, sample_fn, x, MersenneTwister(7);
            n_samples_per_member=n_samples_per)

        # Total samples = n_members * n_samples_per_member
        @test size(samples) == (param_dim, n_members * n_samples_per)
        @test all(isfinite, samples)
    end

    @testset "Higher disagreement for out-of-distribution inputs" begin
        n_members = 3

        # Train on data in [0, 1] range
        rng_data = MersenneTwister(1)
        N = 128
        true_params = rand(rng_data, Float32, param_dim, N)  # in [0,1]
        W = randn(rng_data, Float32, obs_dim, param_dim)
        signals = W * true_params

        build_fn = () -> begin
            model = build_mdn(; obs_dim=obs_dim, param_dim=param_dim,
                n_components=n_components, hidden_dim=hidden_dim, depth=depth)
            rng_init = MersenneTwister(rand(UInt32))
            ps, st = Lux.setup(rng_init, model)
            return (model, ps, st)
        end

        train_fn = (model, ps, st, seed) -> begin
            rng_t = MersenneTwister(seed)
            ps, st, losses = train_mdn!(model, ps, st, true_params, signals, rng_t;
                n_epochs=30, batch_size=32, lr=1e-3,
                n_components=n_components, param_dim=param_dim)
            return (ps, st, losses)
        end

        ensemble = train_ensemble(build_fn, train_fn, n_members; seeds=[100, 200, 300])

        predict_fn = (model, ps, st, x) -> begin
            pi_w, mu, _, _ = mdn_forward(model, ps, st, x;
                n_components=n_components, param_dim=param_dim)
            return dropdims(sum(pi_w' .* mu[:, :, 1], dims=2), dims=2)
        end

        # In-distribution input
        x_id = W * rand(MersenneTwister(50), Float32, param_dim, 1)
        preds_id = ensemble_predict(ensemble, predict_fn, x_id)
        std_id = ensemble_std(preds_id)

        # Out-of-distribution input (very far from training range)
        x_ood = W * (100.0f0 .* ones(Float32, param_dim, 1))
        preds_ood = ensemble_predict(ensemble, predict_fn, x_ood)
        std_ood = ensemble_std(preds_ood)

        # OOD should have higher disagreement on at least one dimension
        @test maximum(std_ood) > maximum(std_id)
    end

    @testset "Integration: 3-member ensemble on toy data, mean better than single" begin
        n_members = 3

        # 1D toy problem: param -> signal = 2*param + 1 + noise
        obs_d = 1
        par_d = 1
        n_comp = 2

        rng_data = MersenneTwister(42)
        N = 256
        true_params = randn(rng_data, Float32, par_d, N)
        signals = 2.0f0 .* true_params .+ 1.0f0 .+ 0.1f0 .* randn(rng_data, Float32, obs_d, N)

        # Hold out test set
        N_test = 50
        rng_test = MersenneTwister(99)
        test_params = randn(rng_test, Float32, par_d, N_test)
        test_signals = 2.0f0 .* test_params .+ 1.0f0

        build_fn = () -> begin
            model = build_mdn(; obs_dim=obs_d, param_dim=par_d,
                n_components=n_comp, hidden_dim=32, depth=2)
            rng_init = MersenneTwister(rand(UInt32))
            ps, st = Lux.setup(rng_init, model)
            return (model, ps, st)
        end

        train_fn = (model, ps, st, seed) -> begin
            rng_t = MersenneTwister(seed)
            ps, st, losses = train_mdn!(model, ps, st, true_params, signals, rng_t;
                n_epochs=80, batch_size=32, lr=1e-3,
                n_components=n_comp, param_dim=par_d)
            return (ps, st, losses)
        end

        ensemble = train_ensemble(build_fn, train_fn, n_members; seeds=[1, 2, 3])

        predict_fn = (model, ps, st, x) -> begin
            pi_w, mu, _, _ = mdn_forward(model, ps, st, x;
                n_components=n_comp, param_dim=par_d)
            return dropdims(sum(pi_w' .* mu[:, :, 1], dims=2), dims=2)
        end

        # Evaluate ensemble mean vs single model on each test point
        ensemble_errors = Float32[]
        single_errors = Float32[]

        for i in 1:N_test
            x_i = test_signals[:, i:i]
            true_p = test_params[:, i]

            preds = ensemble_predict(ensemble, predict_fn, x_i)
            ens_pred = ensemble_mean(preds)
            push!(ensemble_errors, sum((ens_pred .- true_p).^2))

            # Single model prediction (first member only)
            single_pred = predict_fn(ensemble[1][1], ensemble[1][2], ensemble[1][3], x_i)
            push!(single_errors, sum((single_pred .- true_p).^2))
        end

        ensemble_rmse = sqrt(mean(ensemble_errors))
        single_rmse = sqrt(mean(single_errors))

        # Ensemble mean should be at least as good as single model (usually better)
        @test ensemble_rmse <= single_rmse * 1.1  # allow 10% tolerance
        # Both should be reasonably small for this toy problem
        @test ensemble_rmse < 1.0
    end
end
