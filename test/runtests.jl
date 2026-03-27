using Test

using DMI

@testset "DMI.jl" begin
    include("test_compartments.jl")
    include("test_composition.jl")
    include("test_constraints.jl")
    include("test_fitting.jl")
    include("test_dmipy_crossval.jl")
    include("test_distributions.jl")
    include("test_analytical.jl")
    include("test_physics.jl")
    include("test_surrogate.jl")
    include("test_score_posterior.jl")
    include("test_mcmr_generator.jl")
    include("test_film_network.jl")
    include("test_diffusion_field.jl")
    include("test_sphere_gpd.jl")
    include("test_restricted.jl")
    include("test_noddi.jl")
    include("test_mdn.jl")
    include("test_ensemble.jl")
    include("test_mcmc.jl")
    include("test_variational.jl")
    include("test_sbc.jl")
    include("test_conformal.jl")
    include("test_ood_ppc.jl")
    include("test_augmentation.jl")
end
