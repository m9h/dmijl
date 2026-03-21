module Microstructure

using Lux, Random, Statistics, ComponentArrays, Optimisers, Zygote

include("models/ball_stick.jl")
include("noise.jl")
include("diffusion/schedule.jl")
include("diffusion/score_net.jl")
include("diffusion/train.jl")
include("diffusion/sample.jl")

export BallStickModel, simulate, add_rician_noise
export VPSchedule, alpha_bar, noise_and_signal
export build_score_net
export train_score!, sample_posterior

end # module
