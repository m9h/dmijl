using Test

using DMI

@testset "DMI.jl" begin
    include("test_compartments.jl")
    include("test_composition.jl")
    include("test_constraints.jl")
    include("test_fitting.jl")
    include("test_dmipy_crossval.jl")
    include("test_analytical.jl")
    include("test_physics.jl")
    include("test_surrogate.jl")
    include("test_score_posterior.jl")
    include("test_mcmr_generator.jl")
    include("test_film_network.jl")
    include("test_diffusion_field.jl")
end
