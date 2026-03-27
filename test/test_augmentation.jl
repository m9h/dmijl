using Test, Random, LinearAlgebra, Statistics

@testset "Data Augmentation" begin

    # ---- Variable SNR Noise ----

    @testset "add_variable_snr_noise" begin
        rng = MersenneTwister(42)
        signals = ones(Float64, 90, 100)  # (n_meas, n_samples)

        @testset "output shape matches input" begin
            noisy = DMI.add_variable_snr_noise(signals, rng; snr_range=(10.0, 50.0))
            @test size(noisy) == size(signals)
        end

        @testset "SNR range respected — noise level varies across samples" begin
            rng2 = MersenneTwister(123)
            clean = ones(Float64, 90, 500)
            noisy = DMI.add_variable_snr_noise(clean, rng2; snr_range=(10.0, 50.0))
            # For unit signals with Rician noise, the standard deviation across
            # measurements for a given sample reflects sigma.
            # Different columns should have different noise levels.
            col_stds = [std(noisy[:, j]) for j in 1:500]
            # The range of per-column stds should be non-trivial (not all equal)
            @test maximum(col_stds) > 1.5 * minimum(col_stds)
        end

        @testset "signal stays non-negative (Rician-like)" begin
            rng3 = MersenneTwister(999)
            sig = rand(MersenneTwister(0), 90, 200)  # random positive signals
            noisy = DMI.add_variable_snr_noise(sig, rng3; snr_range=(5.0, 100.0))
            @test all(noisy .>= 0.0)
        end
    end

    # ---- B0 Normalization ----

    @testset "normalize_b0" begin
        @testset "divides by b=0 signal, output b=0 entries approx 1" begin
            bvalues = vcat(zeros(6), fill(1e9, 30), fill(2e9, 30))
            # Create signals where b=0 entries have value 5.0
            signals = ones(Float64, 66, 10)
            signals[1:6, :] .= 5.0  # b=0 volumes
            signals[7:end, :] .= 2.5  # DW volumes

            normed = DMI.normalize_b0(signals, bvalues)
            @test size(normed) == size(signals)
            # b=0 entries should be approximately 1.0
            @test all(abs.(normed[1:6, :] .- 1.0) .< 1e-10)
            # DW entries should be 2.5 / 5.0 = 0.5
            @test all(abs.(normed[7:end, :] .- 0.5) .< 1e-10)
        end

        @testset "handles multiple b=0 volumes with different values" begin
            bvalues = vcat(zeros(3), fill(1e9, 5))
            signals = zeros(Float64, 8, 4)
            # b=0 volumes with different values per sample
            signals[1, :] = [4.0, 6.0, 8.0, 10.0]
            signals[2, :] = [6.0, 4.0, 12.0, 10.0]
            signals[3, :] = [5.0, 5.0, 10.0, 10.0]
            signals[4:8, :] .= 2.0

            normed = DMI.normalize_b0(signals, bvalues)
            # Mean b0 per sample: [5.0, 5.0, 10.0, 10.0]
            @test normed[4, 1] ≈ 2.0 / 5.0 atol=1e-10
            @test normed[4, 3] ≈ 2.0 / 10.0 atol=1e-10
            # b=0 entries should average to 1.0 per sample
            for j in 1:4
                @test mean(normed[1:3, j]) ≈ 1.0 atol=1e-10
            end
        end
    end

    # ---- Label Switching Fix ----

    @testset "fix_label_switching" begin
        # param_layout describes which indices are volume fractions and
        # which are corresponding orientation parameters
        # For Ball+2Stick: params = [d_ball, d_stick, f1, f2, mu1x, mu1y, mu1z, mu2x, mu2y, mu2z]
        layout = DMI.FiberLayout(
            fraction_indices = [3, 4],
            orientation_indices = [5:7, 8:10]
        )

        @testset "ensures f1 >= f2 (canonical ordering)" begin
            # 5 samples, 10 params each
            params = zeros(Float64, 10, 5)
            params[1, :] .= 1.7e-9  # d_ball
            params[2, :] .= 1.0e-9  # d_stick
            # Sample 1: f1 < f2 (needs swap)
            params[3, 1] = 0.1; params[4, 1] = 0.4
            # Sample 2: f1 > f2 (already correct)
            params[3, 2] = 0.5; params[4, 2] = 0.2
            # Sample 3: f1 == f2 (no swap needed)
            params[3, 3] = 0.3; params[4, 3] = 0.3

            fixed = DMI.fix_label_switching(params, layout)
            # After fixing, f1 >= f2 for all samples
            for j in 1:5
                @test fixed[3, j] >= fixed[4, j]
            end
        end

        @testset "swaps corresponding orientation vectors when reordering" begin
            params = zeros(Float64, 10, 2)
            params[1, :] .= 1.7e-9
            params[2, :] .= 1.0e-9
            # Sample 1: f1 < f2 => needs swap
            params[3, 1] = 0.1; params[4, 1] = 0.4
            params[5:7, 1] = [1.0, 0.0, 0.0]  # mu1
            params[8:10, 1] = [0.0, 1.0, 0.0]  # mu2
            # Sample 2: f1 > f2 => no swap
            params[3, 2] = 0.5; params[4, 2] = 0.2
            params[5:7, 2] = [1.0, 0.0, 0.0]
            params[8:10, 2] = [0.0, 1.0, 0.0]

            fixed = DMI.fix_label_switching(params, layout)
            # Sample 1 should have swapped: mu1 is now [0,1,0], mu2 is [1,0,0]
            @test fixed[5:7, 1] ≈ [0.0, 1.0, 0.0]
            @test fixed[8:10, 1] ≈ [1.0, 0.0, 0.0]
            @test fixed[3, 1] ≈ 0.4
            @test fixed[4, 1] ≈ 0.1
            # Sample 2 should be unchanged
            @test fixed[5:7, 2] ≈ [1.0, 0.0, 0.0]
            @test fixed[8:10, 2] ≈ [0.0, 1.0, 0.0]
            @test fixed[3, 2] ≈ 0.5
            @test fixed[4, 2] ≈ 0.2
        end
    end

    # ---- Full Augmentation Pipeline ----

    @testset "augment_training_batch" begin
        rng = MersenneTwister(7)
        n_meas = 66
        n_samples = 50
        bvalues = vcat(zeros(6), fill(1e9, 30), fill(2e9, 30))
        signals = rand(MersenneTwister(1), n_meas, n_samples) .* 0.5 .+ 0.5
        params = rand(MersenneTwister(2), 10, n_samples)

        @testset "combines all augmentations in pipeline" begin
            p_aug, s_aug = DMI.augment_training_batch(
                params, signals, bvalues, rng;
                snr_range=(10.0, 50.0), normalize=true,
                fix_switching=false, param_layout=nothing
            )
            # Signals should be noisy and normalized
            @test !isapprox(s_aug, signals, atol=0.01)
            # b=0 entries should be near 1.0 (normalized)
            b0_vals = s_aug[1:6, :]
            @test mean(b0_vals) > 0.5  # should be around 1.0 after normalization
        end

        @testset "output shapes match input" begin
            p_aug, s_aug = DMI.augment_training_batch(
                params, signals, bvalues, rng;
                snr_range=(10.0, 50.0), normalize=true
            )
            @test size(p_aug) == size(params)
            @test size(s_aug) == (n_meas, n_samples)
        end
    end

    # ---- Integration Test ----

    @testset "Integration: Ball+2Stick augmentation" begin
        rng = MersenneTwister(314)

        # Build acquisition
        bvalues = vcat(zeros(6), fill(1e9, 30), fill(2e9, 30))
        bvecs_b0 = repeat([1.0 0.0 0.0], 6, 1)
        rng_vecs = MersenneTwister(0)
        function rand_unit_vecs(rng, n)
            z = randn(rng, n, 3)
            z ./ sqrt.(sum(z.^2, dims=2))
        end
        bvecs_dw = rand_unit_vecs(rng_vecs, 60)
        bvecs = vcat(bvecs_b0, bvecs_dw)
        model = BallStickModel(bvalues, bvecs)

        # Generate clean Ball+2Stick data
        n_samples = 100
        theta = zeros(10, n_samples)
        for j in 1:n_samples
            theta[1, j] = 1.7e-9                  # d_ball
            theta[2, j] = 1.0e-9                   # d_stick
            f1 = rand(rng) * 0.3
            f2 = rand(rng) * 0.3
            theta[3, j] = f1
            theta[4, j] = f2
            mu1 = randn(rng, 3); mu1 /= norm(mu1)
            mu2 = randn(rng, 3); mu2 /= norm(mu2)
            theta[5:7, j] = mu1
            theta[8:10, j] = mu2
        end

        # Simulate signals (column-major: n_meas x n_samples)
        signals = zeros(66, n_samples)
        for j in 1:n_samples
            signals[:, j] = simulate(model, theta[:, j])
        end

        layout = DMI.FiberLayout(
            fraction_indices = [3, 4],
            orientation_indices = [5:7, 8:10]
        )

        # Augment
        p_aug, s_aug = DMI.augment_training_batch(
            theta, signals, bvalues, rng;
            snr_range=(15.0, 40.0), normalize=true,
            fix_switching=true, param_layout=layout
        )

        # Verify properties preserved
        @test size(s_aug) == (66, n_samples)
        @test size(p_aug) == (10, n_samples)
        # All augmented signals should be non-negative
        @test all(s_aug .>= 0.0)
        # After label switching fix, f1 >= f2
        for j in 1:n_samples
            @test p_aug[3, j] >= p_aug[4, j]
        end
        # b=0 values should be close to 1.0 after normalization
        b0_mean_per_sample = mean(s_aug[1:6, :], dims=1)
        @test all(abs.(b0_mean_per_sample .- 1.0) .< 0.5)  # with noise, allow tolerance
    end
end
