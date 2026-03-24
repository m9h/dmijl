"""
Microstructure.jl — Score-based dMRI microstructure estimation in Julia.

Focuses on what Julia does better than Python/JAX:
- Native SDE solvers (DifferentialEquations.jl) for reverse diffusion
- No XLA compilation wall → fast autoresearch iteration
- MCMRSimulator.jl integration for Monte Carlo forward simulation
- Multiple forward models with Julia's multiple dispatch
"""
module Microstructure

using Lux, Random, Statistics, LinearAlgebra
using ComponentArrays, Optimisers, Zygote

# ---- Acquisition ----
include("pipeline/acquisition.jl")
export Acquisition, hcp_like_acquisition, load_acquisition

# ---- Forward models ----
include("models/ball_stick.jl")
export BallStickModel

include("models/dti.jl")
export DTIModel, compute_fa, compute_md, compute_ad, compute_rd

include("models/noddi.jl")
export NODDIModel, kappa_to_odi

# ---- Noise ----
include("noise.jl")
export add_rician_noise

# ---- Pipeline ----
include("pipeline/config.jl")
export SBIConfig

include("pipeline/simulator.jl")
export ModelSimulator, sample_prior, simulate, add_noise, sample_and_simulate

# ---- Score-based diffusion posterior ----
include("diffusion/schedule.jl")
export VPSchedule, alpha_bar, noise_and_signal

include("diffusion/score_net.jl")
export build_score_net, score_forward

include("diffusion/train.jl")
export train_score!

include("diffusion/sample.jl")
export sample_posterior

include("diffusion/sample_diffeq.jl")
export sample_posterior_diffeq, sample_posterior_ode

# ---- Bloch-Torrey neural surrogate / PINN ----
include("pinn/bloch_torrey.jl")
export build_surrogate, train_surrogate!, BlochTorreyResidual

# ---- Surrogate-accelerated SBI pipeline ----
include("pipeline/surrogate_sbi.jl")
export train_surrogate_sbi

# ---- Evaluation ----
include("validation/metrics.jl")
export angular_error_deg, pearson_r, rmse, evaluate_ball2stick

end # module
