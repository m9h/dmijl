using Test
using DMI
using LinearAlgebra

@testset "FEM Bloch-Torrey" begin

    @testset "Build FEM cylinder" begin
        R = 3e-6   # 3 µm
        D = 1.7e-9 # m²/s
        geom = build_fem_cylinder(R, D; neig=30)

        @test geom isa FEMGeometry
        @test geom.R_um ≈ 3.0  # µm
        @test geom.D_um ≈ 1.7e-3  # µm²/µs
    end

    @testset "FEM signal — basic properties" begin
        R = 3e-6
        D = 1.7e-9
        geom = build_fem_cylinder(R, D; neig=30)

        delta = 10e-3  # 10 ms
        Delta = 30e-3  # 30 ms
        dir = [1.0, 0.0, 0.0]

        # b=0 → signal = 1
        S0 = fem_signal(geom, delta, Delta, 0.0, dir)
        @test S0 ≈ 1.0

        # Signal should decrease with b
        S_low = fem_signal(geom, delta, Delta, 1e9, dir)
        S_high = fem_signal(geom, delta, Delta, 4e9, dir)
        @test S_low < 1.0
        @test S_high < S_low
        @test S_low > 0.0
        @test S_high > 0.0

        # Signal should be in [0, 1]
        @test 0 <= S_low <= 1
        @test 0 <= S_high <= 1
    end

    @testset "FEM vs Van Gelderen — short pulse limit" begin
        # For very short δ, FEM and Van Gelderen should agree closely
        R = 3e-6
        D = 1.7e-9
        delta = 1e-3  # 1 ms — short pulse
        Delta = 30e-3
        bvalues = Float64[0, 1e9, 2e9, 4e9]
        dirs = repeat([1.0 0.0 0.0], 4)

        S_fem = fem_cylinder_signal(R, D, delta, Delta, bvalues, dirs; neig=30)
        S_vg = [b < 1e3 ? 1.0 : van_gelderen_cylinder(R, D, delta, Delta, b) for b in bvalues]

        # Should agree within ~10% for short pulses
        for i in 2:length(bvalues)
            rel_err = abs(S_fem[i] - S_vg[i]) / max(S_vg[i], 1e-10)
            @test rel_err < 0.15  # 15% tolerance (FEM mesh is coarse with neig=30)
        end
    end

    @testset "FEM multi-compartment AxCaliber signal" begin
        R = 3e-6
        D_intra = 1.7e-9
        D_extra = 1.0e-9
        f = 0.5
        mu = [0.0, 0.0, 1.0]

        bvals = Float64[0, 1e9, 2e9]
        dirs = [1.0 0.0 0.0; 1.0 0.0 0.0; 0.0 0.0 1.0]
        acq = Acquisition(bvals, dirs, 10e-3, 30e-3)

        S = fem_axcaliber_signal(R, D_intra, D_extra, f, mu, acq; neig=20)
        @test length(S) == 3
        @test S[1] ≈ 1.0
        @test all(0 .<= S .<= 1.0)

        # Parallel to fiber axis should show less restriction
        # (gradient along z, fiber along z → mostly free diffusion)
        @test S[3] > S[2]  # parallel > perpendicular attenuation recovery
    end
end
