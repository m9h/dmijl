using Test

@testset "Microstructure.jl" begin
    include("test_analytical.jl")
    include("test_physics.jl")
    include("test_surrogate.jl")
    include("test_score_posterior.jl")
    include("test_mcmr_generator.jl")
    include("test_film_network.jl")
end
