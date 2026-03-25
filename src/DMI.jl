"""
DMI.jl — Diffusion Microstructural Imaging in Julia.

Focuses on what Julia does better than Python/JAX:
- Native SDE solvers (DifferentialEquations.jl) for reverse diffusion
- No XLA compilation wall → fast autoresearch iteration
- MCMRSimulator.jl integration for Monte Carlo forward simulation
- Multiple forward models with Julia's multiple dispatch
"""
module DMI

using Lux, Random, Statistics, LinearAlgebra
using ComponentArrays, Optimisers, Zygote

# ---- GPU / device utilities ----
include("gpu.jl")
export select_device, to_device

# ---- Acquisition ----
include("pipeline/acquisition.jl")
export Acquisition, hcp_like_acquisition, load_acquisition

# ---- Composable compartment models ----
include("compartments/types.jl")
export AbstractCompartment, signal, parameter_names, parameter_ranges, parameter_cardinality, nparams

include("compartments/ball.jl")
export G1Ball

include("compartments/stick.jl")
export C1Stick

include("compartments/zeppelin.jl")
export G2Zeppelin

include("compartments/dot.jl")
export S1Dot

# ---- Multi-compartment composition ----
include("composition/multi_compartment.jl")
export MultiCompartmentModel
export parameter_dictionary_to_array, parameter_array_to_dictionary, get_flat_bounds

# ---- Parameter constraints ----
include("composition/constraints.jl")
export ConstrainedModel, set_fixed_parameter, set_volume_fraction_unity, set_tortuosity

# ---- Multi-compartment fitting ----
include("fitting/nlls.jl")
export fit_mcm, fit_mcm_batch

# ---- Legacy forward models ----
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
export SinusoidalEmbedding, FiLMLayer, ConditioningNet, ScoreNetwork
export build_score_net, score_forward

include("diffusion/train.jl")
export train_score!

include("diffusion/sample.jl")
export sample_posterior

include("diffusion/sample_diffeq.jl")
export sample_posterior_diffeq, sample_posterior_ode

# ---- Bloch-Torrey neural surrogate / PINN ----
include("pinn/bloch_torrey.jl")
export build_surrogate, train_surrogate!, BlochTorreyResidual, pde_loss, train_pinn!

# ---- AxCaliber PINN (proper physics-informed with Van Gelderen) ----
include("pinn/axcaliber_pinn.jl")
export van_gelderen_cylinder, axcaliber_signal
export AxCaliberData, build_axcaliber_pinn, train_axcaliber_pinn!, decode_geometry

# ---- Non-parametric diffusion field recovery ----
include("pinn/diffusion_field.jl")
export DiffusionFieldProblem, solve_diffusion_field, extract_maps
export build_diffusivity_net, build_magnetization_net, eval_D

# ---- Direction-aware diffusion field recovery (v2) ----
include("pinn/diffusion_field_v2.jl")
export predict_signal_directional, solve_diffusion_field_v2

# ---- MCMRSimulator training data generation ----
include("pipeline/mcmr_generator.jl")
export MCMRGeometry, generate_mcmr_training_data,
       sample_cylinder_geometry, sample_sphere_geometry,
       mcmr_data_fn

# ---- Surrogate-accelerated SBI pipeline ----
include("pipeline/surrogate_sbi.jl")
export train_surrogate_sbi

# ---- Evaluation ----
include("validation/metrics.jl")
export angular_error_deg, pearson_r, rmse, evaluate_ball2stick

# ---- Compatibility with Ting Gong's Microstructure.jl ----
include("compat/microstructure_jl.jl")
export MicrostructureProtocol, load_protocol, protocol_from_bval_bvec
export cross_validate_compartments, load_for_dfield

include("validation/koma_oracle.jl")
export validate_free_diffusion_koma, validate_signal_properties_koma

end # module
