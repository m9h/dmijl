# API Reference

## Compartment Models

### Types
- `AbstractCompartment` — base type for all compartments
- `signal(compartment, acq, params)` — compute signal attenuation
- `parameter_names(compartment)` — ordered parameter name list
- `parameter_ranges(compartment)` — `Dict{String, Tuple{Float64, Float64}}`
- `nparams(compartment)` — number of parameters

### Compartments
- `G1Ball()` — isotropic Gaussian (1 param: lambda_iso)
- `C1Stick()` — intra-axonal stick (4 params: lambda_par, mu_xyz)
- `G2Zeppelin()` — axially symmetric tensor (5 params: lambda_par, lambda_perp, mu_xyz)
- `S1Dot()` — stationary water (0 params)

### Composition
- `MultiCompartmentModel(compartments)` — combine with volume fractions
- `parameter_dictionary_to_array(mcm, dict)` — dict to flat vector
- `parameter_array_to_dictionary(mcm, array)` — flat vector to dict
- `get_flat_bounds(mcm)` — (lows, highs) for all parameters

### Constraints
- `ConstrainedModel(mcm)` — wrap MCM with parameter constraints
- `set_fixed_parameter(cm, name, value)` — fix a parameter
- `set_volume_fraction_unity(cm)` — fractions sum to 1
- `set_tortuosity(cm, perp, par, fraction)` — tortuosity constraint

### Orientation Distributions
- `WatsonDistribution(; n_grid=300)` — Watson distribution on Fibonacci grid
- `watson_weights(watson, kappa, mu)` — compute weights for given κ, μ
- `DistributedModel(compartment, watson)` — convolve compartment with Watson ODF

### Fitting
- `fit_mcm(mcm, acq, signal; n_restarts)` — NLLS fit single voxel
- `fit_mcm_batch(mcm, acq, signals; n_restarts)` — batch fitting

## Legacy Forward Models

### Ball+2Stick
- `BallStickModel(bvalues, gradient_directions)` — construct model
- `simulate(model, params)` — compute signal for a parameter vector

### DTI
- `DTIModel(bvalues, gradient_directions)` — construct model
- `compute_fa(λ1, λ2, λ3)`, `compute_md`, `compute_ad`, `compute_rd`

### NODDI
- `NODDIModel(bvalues, gradient_directions)` — construct model
- `kappa_to_odi(kappa)` — Watson κ to ODI conversion

### Van Gelderen (restricted diffusion)
- `van_gelderen_cylinder(b, delta, Delta, D, R)` — cylinder signal attenuation
- `axcaliber_signal(b, delta, Delta, D_intra, D_extra, R, f, g, mu)` — multi-compartment

## AxCaliber PINN

- `AxCaliberData(signals, bvalues, bvecs, deltas, Deltas)` — multi-delta data container
- `build_axcaliber_pinn(; signal_dim, hidden_dim, depth)` — construct network
- `train_axcaliber_pinn!(model, ps, st, data; n_steps, lambda_physics)` — train
- `decode_geometry(raw_output)` — convert network output to physical parameters

## Diffusion Tensor Field

- `DiffusionFieldProblem(signal, bvals, bvecs, delta, Delta, T2, voxel_size)`
- `build_diffusivity_net(; hidden_dim, depth, output_type)` — D-network
- `eval_D(net, ps, st, x, output_type)` — evaluate D at position x
- `solve_diffusion_field_v2(problem; ...)` — direction-aware recovery
- `extract_maps(result; grid_resolution)` — compute FA, MD from D-field

## Score-Based Posterior

- `build_score_net(; param_dim, signal_dim, hidden_dim, depth, cond_dim)` — score network
- `VPSchedule(beta_min, beta_max)` — noise schedule
- `train_score!(model, ps, st; simulator_fn, prior_fn, schedule, ...)` — train
- `sample_posterior(model, ps, st, signal; n_samples, n_steps, ...)` — DDPM sampling
- `sample_posterior_diffeq(score_fn, signal, schedule; solver, ...)` — DiffEq SDE
- `sample_posterior_ode(score_fn, signal, schedule; solver, ...)` — probability flow ODE

## Pipeline

- `Acquisition(bvalues, gradient_directions)` — acquisition spec
- `hcp_like_acquisition()` — 90-direction multi-shell
- `load_acquisition(bval_path, bvec_path)` — FSL format loading
- `SBIConfig(...)` — pipeline configuration
- `ModelSimulator(forward_fn, names, ranges, bvals, bvecs; ...)` — simulator wrapper
- `sample_and_simulate(sim, rng, n)` — generate training data

## Surrogate

- `build_surrogate(; param_dim, signal_dim, hidden_dim, depth)` — MLP surrogate
- `train_surrogate!(model, ps, st, data_fn; n_steps, loss_type, ...)` — train
- `BlochTorreyResidual(; gradient_fn)` — PDE residual specification
- `pde_loss(residual, model, ps, st, t, x, D, T2)` — evaluate PDE residual
- `train_pinn!(model, ps, st, data_fn, residual; lambda_pde, ...)` — combined training

## Validation

- `angular_error_deg(mu_true, mu_pred)` — angle between unit vectors
- `pearson_r(x, y)` — Pearson correlation
- `rmse(pred, true_val)` — root mean squared error
- `evaluate_ball2stick(theta_true, theta_pred)` — full Ball+2Stick evaluation
- `cross_validate_compartments()` — validate against Microstructure.jl
- `validate_signal_properties_koma(model)` — validate against KomaMRI

## Compatibility

- `MicrostructureProtocol(bval, techo, tdelta, tsmalldel, gvec)` — Microstructure.jl format
- `load_protocol(btable_file)` — load .btable
- `protocol_from_bval_bvec(bval_file, bvec_file; ...)` — FSL format
- `load_for_dfield(image_file, btable_file; ...)` — load data for D(r) recovery

## GPU

- `select_device()` — auto-detect CUDA GPU, fallback to CPU
- `to_device(x, dev)` — transfer arrays to device
