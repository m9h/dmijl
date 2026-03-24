"""
Tests for MCMRGeometry construction and geometry sampler functions.

Does NOT call MCMRSimulator (those are slow integration tests).
Tests the data structures, validation, and sampler distributions only.
"""

using Test, Random, LinearAlgebra

# Include source directly (before package is registered)
include("../src/pipeline/mcmr_generator.jl")

@testset "MCMRGenerator" begin

    @testset "MCMRGeometry construction (cylinders)" begin
        geo = MCMRGeometry(
            geometry_type = :cylinders,
            mean_radius = 1.5,
            radius_variance = 0.2,
            volume_fraction = 0.6,
            box_size = 20.0,
        )
        @test geo.geometry_type == :cylinders
        @test geo.mean_radius == 1.5
        @test geo.radius_variance == 0.2
        @test geo.volume_fraction == 0.6
        @test geo.box_size == 20.0
        @test geo.rotation === nothing
    end

    @testset "MCMRGeometry construction (spheres)" begin
        geo = MCMRGeometry(
            geometry_type = :spheres,
            mean_radius = 5.0,
            radius_variance = 1.0,
            volume_fraction = 0.5,
        )
        @test geo.geometry_type == :spheres
        @test geo.mean_radius == 5.0
        @test geo.radius_variance == 1.0
        @test geo.volume_fraction == 0.5
    end

    @testset "MCMRGeometry with rotation" begin
        R = Matrix{Float64}(I, 3, 3)
        geo = MCMRGeometry(
            geometry_type = :cylinders,
            mean_radius = 2.0,
            radius_variance = 0.3,
            volume_fraction = 0.7,
            rotation = R,
        )
        @test geo.rotation !== nothing
        @test geo.rotation == R
    end

    @testset "MCMRGeometry validation errors" begin
        # Invalid geometry type
        @test_throws AssertionError MCMRGeometry(geometry_type = :cubes)

        # Negative radius
        @test_throws AssertionError MCMRGeometry(mean_radius = -1.0)

        # Negative radius variance
        @test_throws AssertionError MCMRGeometry(radius_variance = -0.1)

        # Volume fraction out of bounds
        @test_throws AssertionError MCMRGeometry(volume_fraction = 0.0)
        @test_throws AssertionError MCMRGeometry(volume_fraction = 0.9)

        # Negative box size
        @test_throws AssertionError MCMRGeometry(box_size = -5.0)
    end

    @testset "param_names and param_dim" begin
        geo_cyl = MCMRGeometry(geometry_type = :cylinders)
        geo_sph = MCMRGeometry(geometry_type = :spheres)

        @test param_dim(geo_cyl) == 3
        @test param_dim(geo_sph) == 3

        names_cyl = param_names(geo_cyl)
        names_sph = param_names(geo_sph)
        @test length(names_cyl) == 3
        @test length(names_sph) == 3
        @test "mean_radius" in names_cyl
        @test "radius_variance" in names_cyl
        @test "volume_fraction" in names_cyl
    end

    @testset "sample_cylinder_geometry" begin
        rng = MersenneTwister(42)
        params, geo = sample_cylinder_geometry(rng)

        # Params are Float32 vector of length 3
        @test length(params) == 3
        @test eltype(params) == Float32

        # Params match geometry fields
        @test params[1] ≈ Float32(geo.mean_radius) atol=1e-5
        @test params[2] ≈ Float32(geo.radius_variance) atol=1e-5
        @test params[3] ≈ Float32(geo.volume_fraction) atol=1e-5

        # Geometry is cylinders
        @test geo.geometry_type == :cylinders

        # Prior ranges
        @test 0.3 <= geo.mean_radius <= 5.0
        @test 0.01 <= geo.radius_variance <= 0.5
        @test 0.3 <= geo.volume_fraction <= 0.8
    end

    @testset "sample_sphere_geometry" begin
        rng = MersenneTwister(42)
        params, geo = sample_sphere_geometry(rng)

        # Params are Float32 vector of length 3
        @test length(params) == 3
        @test eltype(params) == Float32

        # Geometry is spheres
        @test geo.geometry_type == :spheres

        # Prior ranges
        @test 3.0 <= geo.mean_radius <= 8.0
        @test 0.1 <= geo.radius_variance <= 2.0
        @test 0.2 <= geo.volume_fraction <= 0.7
    end

    @testset "Sampler distributions cover prior range" begin
        rng = MersenneTwister(123)
        n = 200

        cyl_radii = Float64[]
        sph_radii = Float64[]
        cyl_vf = Float64[]
        sph_vf = Float64[]

        for _ in 1:n
            pc, gc = sample_cylinder_geometry(rng)
            ps, gs = sample_sphere_geometry(rng)
            push!(cyl_radii, gc.mean_radius)
            push!(sph_radii, gs.mean_radius)
            push!(cyl_vf, gc.volume_fraction)
            push!(sph_vf, gs.volume_fraction)
        end

        # Check that samples span a reasonable fraction of the prior range
        @test minimum(cyl_radii) < 1.0   # lower end explored
        @test maximum(cyl_radii) > 4.0   # upper end explored
        @test minimum(sph_radii) < 4.0
        @test maximum(sph_radii) > 7.0
        @test minimum(cyl_vf) < 0.4
        @test maximum(cyl_vf) > 0.7
        @test minimum(sph_vf) < 0.3
        @test maximum(sph_vf) > 0.6
    end

    @testset "Different seeds produce different samples" begin
        rng1 = MersenneTwister(1)
        rng2 = MersenneTwister(2)
        p1, _ = sample_cylinder_geometry(rng1)
        p2, _ = sample_cylinder_geometry(rng2)
        @test p1 != p2
    end
end
