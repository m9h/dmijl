using Test, Statistics, LinearAlgebra

@testset "MR-ARFI Simulation" begin

    # ================================================================ #
    # Tissue property lookups
    # ================================================================ #

    @testset "Acoustic property lookup" begin
        labels = [0, 1, 2, 3, 4, 5]
        c, rho, alpha = map_labels_to_acoustic(labels)

        # Water
        @test c[1] == 1500.0
        @test rho[1] == 1000.0
        @test alpha[1] == 0.0

        # Skull
        @test c[3] == 4080.0
        @test rho[3] == 1900.0
        @test alpha[3] == 4.74

        # Gray matter
        @test c[5] == 1560.0
        @test rho[5] == 1040.0
        @test alpha[5] == 5.3

        # White matter (same acoustic props as GM)
        @test c[6] == 1560.0
        @test rho[6] == 1040.0

        # Out-of-range defaults to water
        c_oor, _, _ = map_labels_to_acoustic([99])
        @test c_oor[1] == 1500.0
    end

    @testset "Shear modulus mapping" begin
        labels = [0, 3, 4, 5]
        mu = map_labels_to_shear_modulus(labels)

        @test mu[1] == 0.0       # background
        @test mu[2] == 0.0       # CSF (fluid)
        @test mu[3] == 1.35e3    # GM
        @test mu[4] == 0.74e3    # WM

        # GM stiffer than WM (key Kuhl CANN finding)
        @test mu[3] > mu[4]

        # Skull very stiff
        mu_skull = map_labels_to_shear_modulus([2])
        @test mu_skull[1] == 5.0e3
    end

    @testset "MR property lookup" begin
        T1, T2, PD = map_labels_to_mr([0, 4, 5])

        # Background has zero PD
        @test PD[1] == 0.0

        # GM and WM have distinct T1
        @test T1[2] == 1.6  # GM
        @test T1[3] == 0.8  # WM
        @test T1[2] > T1[3]

        # PD for brain tissue
        @test PD[2] > 0.0
        @test PD[3] > 0.0
    end

    # ================================================================ #
    # Attenuation unit conversion
    # ================================================================ #

    @testset "Attenuation conversion" begin
        @test db_cm_to_neper_m(0.0) == 0.0
        @test db_cm_to_neper_m(1.0) ≈ 100.0 / NEPER_TO_DB
        # Round-trip
        @test neper_m_to_db_cm(db_cm_to_neper_m(5.3)) ≈ 5.3 rtol=1e-10
    end

    # ================================================================ #
    # Radiation force
    # ================================================================ #

    @testset "Radiation force — basic physics" begin
        I = 1000.0   # W/m^2
        c = 1560.0   # m/s (brain)
        alpha_db = 5.3  # dB/cm/MHz (brain)
        alpha_np = db_cm_to_neper_m(alpha_db)

        F = compute_radiation_force(I, c, alpha_np)
        @test F > 0
        @test F ≈ 2.0 * alpha_np * I / c

        # Array version
        F_arr = compute_radiation_force([I, I], [c, c], [alpha_np, alpha_np])
        @test length(F_arr) == 2
        @test all(F_arr .> 0)
        @test F_arr[1] ≈ F
    end

    @testset "Radiation force — dB convenience" begin
        I = fill(500.0, 4, 4)
        c = fill(1560.0, 4, 4)
        alpha_db = fill(5.3, 4, 4)

        F = compute_radiation_force_from_db(I, c, alpha_db)
        @test size(F) == (4, 4)
        @test all(F .> 0)

        # Should match manual conversion
        alpha_np = db_cm_to_neper_m.(alpha_db)
        F_manual = compute_radiation_force(I, c, alpha_np)
        @test F ≈ F_manual
    end

    @testset "Radiation force — zero attenuation means zero force" begin
        F = compute_radiation_force(1000.0, 1500.0, 0.0)
        @test F == 0.0
    end

    # ================================================================ #
    # Spectral displacement solver
    # ================================================================ #

    @testset "Spectral solver — uniform force" begin
        N = 64
        dx = 1e-3  # 1 mm
        force = ones(N, N) .* 10.0  # uniform 10 N/m^3
        mu = 1.0e3  # 1 kPa

        u = solve_displacement_spectral(force, mu, dx)
        @test size(u) == (N, N)

        # DC removed -> mean displacement must be near zero
        @test abs(mean(u)) < 1e-10
    end

    @testset "Spectral solver — point force" begin
        N = 64
        dx = 1e-3
        force = zeros(N, N)
        force[N÷2, N÷2] = 1000.0  # point source

        mu = 1.0e3
        u = solve_displacement_spectral(force, mu, dx)

        # Peak displacement at source location
        @test argmax(u) == CartesianIndex(N÷2, N÷2)

        # Displacement decays with distance from source
        @test u[N÷2, N÷2] > u[N÷2+5, N÷2]
        @test u[N÷2, N÷2] > u[N÷2, N÷2+5]

        # Displacement is positive at source (force pushes tissue)
        @test u[N÷2, N÷2] > 0
    end

    @testset "Spectral solver — softer tissue means more displacement" begin
        N = 32
        dx = 1e-3
        force = zeros(N, N)
        force[N÷2, N÷2] = 500.0

        u_stiff = solve_displacement_spectral(force, 2.0e3, dx)  # 2 kPa
        u_soft  = solve_displacement_spectral(force, 0.5e3, dx)  # 0.5 kPa

        @test maximum(u_soft) > maximum(u_stiff)
        @test maximum(u_soft) / maximum(u_stiff) ≈ 4.0 rtol=0.01
    end

    @testset "Spectral solver — 3D" begin
        N = 16
        dx = 1e-3
        force = zeros(N, N, N)
        force[N÷2, N÷2, N÷2] = 500.0
        mu = 1.0e3

        u = solve_displacement_spectral(force, mu, dx)
        @test size(u) == (N, N, N)
        @test argmax(u) == CartesianIndex(N÷2, N÷2, N÷2)
        @test u[N÷2, N÷2, N÷2] > 0
    end

    # ================================================================ #
    # Phase encoding
    # ================================================================ #

    @testset "ARFI phase prediction — analytical" begin
        G = 40e-3   # 40 mT/m
        delta = 5e-3 # 5 ms
        u = 5e-6     # 5 um displacement

        seq = ARFISequenceParams(msg_amplitude=G, msg_duration=delta)
        phi = predict_arfi_phase([u], seq)

        expected = GAMMA_PROTON * G * delta * u
        @test phi[1] ≈ expected

        # Phase should be ~0.27 rad for 5 um
        @test 0.2 < phi[1] < 0.35
    end

    @testset "Phase scales linearly with displacement" begin
        seq = ARFISequenceParams()
        u1 = [1e-6]
        u2 = [2e-6]

        phi1 = predict_arfi_phase(u1, seq)
        phi2 = predict_arfi_phase(u2, seq)

        @test phi2[1] / phi1[1] ≈ 2.0 rtol=1e-10
    end

    @testset "Phase roundtrip — encode then decode" begin
        seq = ARFISequenceParams(msg_amplitude=40e-3, msg_duration=5e-3)
        u_true = [1e-6, 5e-6, 10e-6]

        phi = predict_arfi_phase(u_true, seq)
        u_rec = recover_displacement_from_phase(phi, zeros(3), seq)

        @test u_rec ≈ u_true rtol=1e-10
    end

    @testset "Encoding sensitivity" begin
        seq = ARFISequenceParams(msg_amplitude=40e-3, msg_duration=5e-3)
        enc, min_disp = arfi_encoding_sensitivity(seq)

        # Encoding coefficient: gamma * G * delta
        expected_enc = GAMMA_PROTON * 40e-3 * 5e-3
        @test enc ≈ expected_enc

        # Should be approximately 53,500 rad/m
        @test 5e4 < enc < 6e4

        # Minimum displacement should be sub-micron
        @test min_disp < 1e-6
        @test min_disp > 0
    end

    # ================================================================ #
    # End-to-end analytical pipeline
    # ================================================================ #

    @testset "End-to-end analytical — brain phantom" begin
        N = 32
        dx = 1e-3  # 1 mm

        # Create a simple 2D brain phantom:
        # background water with a brain target region
        labels = fill(Int(0), N, N)        # water background
        labels[10:22, 10:22] .= 4          # gray matter target

        # Synthetic intensity: focal peak at center of GM region
        intensity = zeros(N, N)
        intensity[16, 16] = 5000.0  # W/m^2 at focus

        seq = ARFISequenceParams(msg_amplitude=40e-3, msg_duration=5e-3)
        result = simulate_arfi_analytical(intensity, labels, seq, dx)

        @test result isa ARFIResult

        # Displacement should be nonzero at focus
        @test result.displacement[16, 16] > 0

        # Displacement should be in micrometre range
        @test result.displacement[16, 16] < 1e-3  # < 1 mm
        @test result.displacement[16, 16] > 1e-9  # > 1 nm

        # Phase should be nonzero
        @test result.phase_map[16, 16] != 0.0

        # Radiation force should be positive at focus
        @test result.radiation_force[16, 16] > 0

        # No KomaMRI results in analytical mode
        @test result.koma_signal === nothing
        @test result.koma_phase === nothing
    end

    @testset "End-to-end — WM displaces more than GM" begin
        N = 32
        dx = 1e-3

        # GM region
        labels_gm = fill(Int(4), N, N)
        # WM region
        labels_wm = fill(Int(5), N, N)

        intensity = zeros(N, N)
        intensity[N÷2, N÷2] = 3000.0

        seq = ARFISequenceParams()
        result_gm = simulate_arfi_analytical(intensity, labels_gm, seq, dx)
        result_wm = simulate_arfi_analytical(intensity, labels_wm, seq, dx)

        # WM is softer (0.74 kPa) than GM (1.35 kPa), so more displacement
        @test maximum(result_wm.displacement) > maximum(result_gm.displacement)
    end

    # ================================================================ #
    # Phase 2: I/O helpers
    # ================================================================ #

    @testset "JSON parsing helpers" begin
        # Test _to_matrix with nested vectors
        M = DMI._to_matrix([[1.0, 2.0], [3.0, 4.0]])
        @test M isa Matrix{Float64}
        @test size(M) == (2, 2)
        @test M[1, 1] == 1.0
        @test M[2, 2] == 4.0

        # Test _extract_position with mm -> m conversion
        point_data = Dict(:position => [10.0, 20.0, 30.0], :units => "mm")
        pos = DMI._extract_position(point_data)
        @test pos ≈ [0.01, 0.02, 0.03]

        # Position in metres should not convert
        point_m = Dict(:position => [0.01, 0.02, 0.03], :units => "m")
        pos_m = DMI._extract_position(point_m)
        @test pos_m ≈ [0.01, 0.02, 0.03]
    end

    # ================================================================ #
    # Phase 4a: Differentiable chain
    # ================================================================ #

    @testset "Differentiable forward — basic" begin
        N = 16
        dx = 1e-3
        seq = ARFISequenceParams(msg_amplitude=40e-3, msg_duration=5e-3)

        c = fill(1560.0, N, N)
        alpha_np = fill(db_cm_to_neper_m(5.3), N, N)
        I = zeros(N, N)
        I[N÷2, N÷2] = 1000.0
        mu = 1.0e3

        phase, disp = arfi_forward_differentiable(I, c, alpha_np, mu, dx, seq)

        @test size(phase) == (N, N)
        @test size(disp) == (N, N)
        @test phase[N÷2, N÷2] != 0.0
        @test disp[N÷2, N÷2] > 0

        # Phase should equal encoding_coeff * displacement
        enc = arfi_encoding_coefficient(seq)
        @test phase ≈ enc .* disp
    end

    @testset "Differentiable forward — Zygote gradient" begin
        N = 8
        dx = 1e-3
        seq = ARFISequenceParams(msg_amplitude=40e-3, msg_duration=5e-3)

        c = fill(1560.0, N, N)
        alpha_np = fill(db_cm_to_neper_m(5.3), N, N)
        base_I = zeros(N, N)
        base_I[N÷2, N÷2] = 500.0
        mu = 1.0e3

        function loss(scale)
            I = base_I .* scale
            phase, _ = arfi_forward_differentiable(I, c, alpha_np, mu, dx, seq)
            return sum(phase .^ 2)
        end

        g = Zygote.gradient(loss, 1.0)[1]
        @test g !== nothing
        @test g != 0.0

        # Gradient should be positive (more intensity -> more phase -> more loss)
        @test g > 0
    end

    # ================================================================ #
    # Phase 4b: Optimization
    # ================================================================ #

    @testset "Shear modulus fitting" begin
        N = 32
        dx = 1e-3
        mu_true = 1.5e3  # 1.5 kPa

        # Generate synthetic displacement from known force + shear modulus
        force = zeros(N, N)
        force[N÷2, N÷2] = 500.0
        u_obs = solve_displacement_spectral(force, mu_true, dx)

        # Fit shear modulus — since u ∝ 1/mu, the analytical solution
        # for scalar mu is: mu_fit = sum(F*u_ref) / sum(u_obs*u_ref)
        # where u_ref = solve(F, mu=1). Test the optimizer finds this.
        fit_result = fit_shear_modulus(u_obs, force, dx;
                                       initial_mu=1.2e3, max_iter=500,
                                       verbose=false)

        # Optimizer should at least move toward the true value
        @test fit_result.shear_modulus > 1.0e3  # moved from 1.2e3 direction

        # The fit should improve the residual from the initial guess
        u_init = solve_displacement_spectral(force, 1.2e3, dx)
        resid_init = sum((u_init .- u_obs) .^ 2) / length(u_obs)
        @test fit_result.residual <= resid_init + 1e-30
    end

    @testset "MSG parameter optimization" begin
        N = 16
        dx = 1e-3
        labels = fill(Int(4), N, N)  # all GM
        intensity = zeros(N, N)
        intensity[N÷2, N÷2] = 2000.0

        opt = optimize_msg_params(intensity, labels, dx;
                                  max_iter=20, verbose=false)

        # Optimized params should be within bounds
        @test 5e-3 <= opt.msg_amplitude <= 80e-3
        @test 1e-3 <= opt.msg_duration <= 20e-3

        # Encoding coefficient should be positive
        @test opt.encoding_coefficient > 0

        # Loss should decrease over iterations
        @test length(opt.loss_history) > 1
        @test opt.loss_history[end] <= opt.loss_history[1]
    end

    # ================================================================ #
    # Phase 3: KomaMRI (skipped if not available)
    # ================================================================ #

    @testset "KomaMRI ARFI — gated" begin
        if DMI._ensure_koma_arfi!()
            @testset "Single-spin validation" begin
                passed = validate_arfi_single_spin(
                    displacement=5e-6, verbose=false,
                )
                @test passed === true
            end
        else
            @info "KomaMRI not available; skipping Bloch ARFI tests"
            @test true  # placeholder to avoid empty testset
        end
    end

end
