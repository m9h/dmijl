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

# ---- Shared utilities ----
include("utils.jl")

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

include("compartments/sphere_gpd.jl")
export SphereGPD

include("compartments/restricted_cylinder.jl")
export RestrictedCylinder

include("compartments/plane.jl")
export PlaneCallaghan

include("compartments/epg.jl")
export EPGCompartment, epg_signal

# ---- Multi-compartment composition ----
include("composition/multi_compartment.jl")
export MultiCompartmentModel
export parameter_dictionary_to_array, parameter_array_to_dictionary, get_flat_bounds

# ---- Parameter constraints ----
include("composition/constraints.jl")
export ConstrainedModel, set_fixed_parameter, set_volume_fraction_unity, set_tortuosity, LinkedParameter

# ---- Multi-compartment fitting ----
include("fitting/nlls.jl")
export fit_mcm, fit_mcm_batch

include("fitting/algebraic_init.jl")
export dti_init, ball_stick_init, noddi_init

# ---- Orientation distributions ----
include("distributions/watson.jl")
export WatsonDistribution, watson_weights, DistributedModel

# ---- Legacy forward models ----
include("models/ball_stick.jl")
export BallStickModel

include("models/dti.jl")
export DTIModel, compute_fa, compute_md, compute_ad, compute_rd

include("models/noddi.jl")
export NODDIModel, kappa_to_odi

include("models/noddi_watson.jl")
export noddi_watson

# ---- Noise ----
include("noise.jl")
export add_rician_noise

# ---- BIDS Data Loaders ----
include("pipeline/bids.jl")
export BIDSSubject, load_dwi, load_megre, load_qmt, find_bids_subjects

# ---- Pipeline ----
include("pipeline/config.jl")
export SBIConfig

include("pipeline/simulator.jl")
export ModelSimulator, sample_prior, simulate, add_noise, sample_and_simulate

include("pipeline/augmentation.jl")
export FiberLayout, add_variable_snr_noise, normalize_b0, fix_label_switching, augment_training_batch

# ---- Phase processing (ROMEO + MriResearchTools) ----
include("pipeline/phase.jl")
export load_multi_echo, robust_brain_mask, correct_bias_field,
       unwrap_phase, voxel_quality, compute_b0, compute_t2star,
       PhaseResult, process_phase, save_phase_result

# ---- Optimal Experimental Design (Phase 3) ----
include("design/types.jl")
export DesignSpace, DesignResult, OEDProblem

include("design/fim.jl")
export jacobian_signal, fisher_information, expected_fim, crlb, rician_fim_correction

include("design/optimality.jl")
export d_optimality, a_optimality, e_optimality, weighted_a_optimality, optimality_criterion

include("design/constraints.jl")
export GYROMAGNETIC_RATIO, max_bvalue, required_gradient, is_feasible,
       electrostatic_directions, compare_protocols

include("design/optimize.jl")
export optimize_protocol

include("design/protocols.jl")
export hcp_protocol, noddi_protocol, axon_diameter_protocol

include("design/bayesian_oed.jl")
export eig_pce, eig_variational, mdn_log_density

include("design/sequential.jl")
export CandidateMeasurement, generate_candidates, sequential_design

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

# ---- Mixture Density Network (amortized inference) ----
include("inference/mdn.jl")
export build_mdn, mdn_forward, mdn_loss, train_mdn!, sample_mdn

# ---- Deep Ensembles ----
include("inference/ensemble.jl")
export train_ensemble, ensemble_predict, ensemble_mean, ensemble_std, ensemble_sample

# ---- MCMC posterior sampling ----
include("inference/mcmc.jl")
export rician_loglikelihood, mcmc_sample, mcmc_summary, log_besseli0

# ---- Variational Inference (amortized approximate posterior) ----
include("inference/variational.jl")
export build_vi_net, vi_forward, elbo_loss, train_vi!, sample_vi

# ---- Bloch-Torrey neural surrogate / PINN ----
include("pinn/bloch_torrey.jl")
export build_surrogate, train_surrogate!, BlochTorreyResidual, pde_loss, train_pinn!

# ---- AxCaliber PINN (proper physics-informed with Van Gelderen) ----
include("pinn/axcaliber_pinn.jl")
export van_gelderen_cylinder, axcaliber_signal
export AxCaliberData, build_axcaliber_pinn, train_axcaliber_pinn!, decode_geometry

# ---- FEM Bloch-Torrey (SpinDoctor-based) ----
include("fem/forward.jl")
export FEMGeometry, build_fem_cylinder, fem_signal, fem_cylinder_signal

include("fem/differentiable.jl")
export FEMSignalCache, build_fem_cache, fem_signal_gradient, fem_axcaliber_signal

include("fem/fit.jl")
export fit_fem_axcaliber

# ---- MR-ARFI Simulation (openlifu + KomaMRI) ----
include("arfi/types.jl")
export AcousticProperties, TissueMRProperties, ARFISequenceParams, TUSSolution, ARFIResult

include("arfi/tissue_properties.jl")
export ACOUSTIC_TABLE, MR_TABLE, SHEAR_TABLE,
       map_labels_to_acoustic, map_labels_to_shear_modulus, map_labels_to_mr,
       db_cm_to_neper_m, neper_m_to_db_cm

include("arfi/radiation_force.jl")
export compute_radiation_force, compute_radiation_force_from_db

include("arfi/displacement.jl")
export solve_displacement_spectral

include("arfi/forward.jl")
export predict_arfi_phase, recover_displacement_from_phase,
       arfi_encoding_coefficient, arfi_encoding_sensitivity,
       simulate_arfi_analytical

include("arfi/io.jl")
export load_tus_solution, compute_radiation_force_from_solution

include("arfi/koma_arfi.jl")
export build_arfi_phantom, build_arfi_sequence,
       simulate_arfi_koma, validate_arfi_single_spin

include("arfi/differentiable.jl")
export arfi_forward_differentiable, arfi_phase_loss, verify_arfi_gradient

include("arfi/fit.jl")
export optimize_msg_params, fit_shear_modulus

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

# ---- OOD Detection ----
include("validation/ood.jl")
export reconstruction_error, mahalanobis_distance, ood_score, ood_detect

# ---- Posterior Predictive Checks ----
include("validation/ppc.jl")
export posterior_predictive_check, ppc_summary

# ---- Compatibility with Ting Gong's Microstructure.jl ----
include("compat/microstructure_jl.jl")
export MicrostructureProtocol, load_protocol, protocol_from_bval_bvec
export cross_validate_compartments, load_for_dfield

include("validation/koma_oracle.jl")
export validate_free_diffusion_koma, validate_signal_properties_koma

include("validation/spindoctor_oracle.jl")
export validate_restricted_cylinder_spindoctor, SpinDoctorValidationResult

# ---- Simulation-Based Calibration ----
include("validation/sbc.jl")
export compute_rank, sbc_ranks, sbc_histogram, sbc_uniformity_test

# ---- Conformal prediction ----
include("validation/conformal.jl")
export split_conformal, cqr_conformal, conformal_coverage

end # module
