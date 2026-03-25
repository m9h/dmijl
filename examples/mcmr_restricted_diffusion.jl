#!/usr/bin/env julia
"""
Score-based posterior estimation for restricted diffusion in white matter.

Uses MCMRSimulator.jl to generate Monte Carlo training data from
packed-cylinder geometries (axon model), then trains a conditional
score network to infer microstructure parameters from dMRI signals.

Pipeline:
  1. Define multi-shell dMRI acquisition (MRIBuilder)
  2. Sample microstructure parameters from prior
  3. Build cylinder geometries and simulate signals (MCMRSimulator)
  4. Optionally train a neural surrogate for speed
  5. Train conditional score network (denoising score matching)
  6. Sample posterior given observed signal
  7. Visualise results (CairoMakie)

Run with:
    julia +1.11 --project=. examples/mcmr_restricted_diffusion.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random, Statistics, LinearAlgebra, Printf
using Lux, Optimisers, Zygote
using MCMRSimulator
using MRIBuilder
using CairoMakie

include("../src/noise.jl")
include("../src/diffusion/schedule.jl")
include("../src/diffusion/score_net.jl")
include("../src/diffusion/train.jl")
include("../src/diffusion/sample.jl")
include("../src/pinn/bloch_torrey.jl")

# ====================================================================
# Configuration
# ====================================================================

# Reproducibility
rng = Random.MersenneTwister(42)

# --- dMRI microstructure parameter space ---
# We estimate 4 scalar parameters (no orientation vectors in this model):
#   1. r      : axon radius               [0.5, 5.0] um
#   2. f      : intra-axonal volume frac   [0.3, 0.8]
#   3. d_par  : parallel diffusivity       [1.0, 2.5] um^2/ms
#   4. d_extra: extra-axonal diffusivity   [0.5, 1.5] um^2/ms

const PARAM_NAMES = ["r", "f", "d_par", "d_extra"]
const PARAM_DIM   = 4
const LOWS  = Float32[0.5, 0.3, 1.0, 0.5]
const HIGHS = Float32[5.0, 0.8, 2.5, 1.5]
const SPANS = HIGHS .- LOWS

# --- Dataset sizes ---
# Production would use 10k-50k samples with 50k-100k spins each.
# This example uses small counts to run in minutes on a laptop.
const N_TRAIN       = 500      # number of training configurations
const N_SPINS       = 5_000    # spins per Monte Carlo simulation
const BOX_SIZE      = 20.0     # repeating geometry tile size (um)

# --- Training hyperparameters ---
const SURROGATE_STEPS = 2_000  # production: 50k+
const SCORE_STEPS     = 3_000  # production: 30k-100k
const BATCH_SIZE      = 64     # production: 256-512
const SCORE_HIDDEN    = 256    # production: 512
const SCORE_DEPTH     = 4      # production: 6

println("=" ^ 70)
println("MCMRSimulator restricted diffusion -- score-based posterior estimation")
println("=" ^ 70)

# ====================================================================
# 1. Define acquisition: HCP-like multi-shell PGSE
# ====================================================================
println("\n--- Step 1: Defining multi-shell dMRI acquisition ---")

# b-values in ms/um^2 (MCMRSimulator / MRIBuilder convention)
# HCP uses b = 1000, 2000, 3000 s/mm^2 = 1.0, 2.0, 3.0 ms/um^2
bvals_shells = [0.0, 1.0, 2.0, 3.0]
n_per_shell  = [6, 10, 10, 8]  # reduced from HCP for speed (6+10+10+8=34)
# Production: use full HCP (6+30+30+24=90 directions)

# Build one DWI sequence per shell/direction combination.
# MCMRSimulator handles each sequence independently in a single Simulation.
# We orient gradients perpendicular to the cylinder axis (z-axis) for
# maximum sensitivity to restricted diffusion.
#
# For the b=0 volumes we still need a sequence (just with bval=0).
# For diffusion-weighted volumes we sample gradient directions in the x-y plane.

function random_perp_directions(rng, n)
    # Random unit vectors in the x-y plane (perpendicular to cylinder axis z)
    angles = rand(rng, n) .* 2pi
    return [[cos(a), sin(a), 0.0] for a in angles]
end

sequences = Sequence[]
bval_list = Float64[]

for (bval, n_dir) in zip(bvals_shells, n_per_shell)
    if bval == 0.0
        for _ in 1:n_dir
            push!(sequences, DWI(bval=0.0, TE=80.0, scanner=Siemens_Prisma))
            push!(bval_list, 0.0)
        end
    else
        dirs = random_perp_directions(rng, n_dir)
        for d in dirs
            push!(sequences, DWI(
                bval=bval, TE=80.0, scanner=Siemens_Prisma,
                gradient=(orientation=d, rise_time=:min),
            ))
            push!(bval_list, bval)
        end
    end
end

const N_MEAS = length(sequences)
const B0_MASK = bval_list .< 0.01

println("  Shells: b = $(bvals_shells) ms/um^2")
println("  Directions per shell: $(n_per_shell)")
println("  Total measurements: $N_MEAS")

# ====================================================================
# 2. Sample microstructure parameters from the prior
# ====================================================================
println("\n--- Step 2: Sampling microstructure parameters ---")

"""
    sample_prior(rng, n) -> Matrix{Float32} (PARAM_DIM, n)

Draw n parameter vectors, normalised to [0, 1].
"""
function sample_prior(rng, n)
    return rand(rng, Float32, PARAM_DIM, n)
end

"""
    denormalise(theta_norm) -> physical parameters

Convert [0,1]-normalised parameters back to physical units.
"""
function denormalise(theta_norm::AbstractVector)
    return LOWS .+ theta_norm .* SPANS
end
function denormalise(theta_norm::AbstractMatrix)
    return LOWS .+ theta_norm .* SPANS
end

theta_demo = sample_prior(rng, 3)
phys_demo  = denormalise(theta_demo)
println("  Example physical parameters:")
for j in 1:3
    @printf("    [%d] r=%.2f um, f=%.2f, d_par=%.2f, d_extra=%.2f um^2/ms\n",
            j, phys_demo[1,j], phys_demo[2,j], phys_demo[3,j], phys_demo[4,j])
end

# ====================================================================
# 3. Generate training data with MCMRSimulator
# ====================================================================
println("\n--- Step 3: Generating training data with MCMRSimulator ---")
println("  $N_TRAIN configurations x $N_SPINS spins each")
println("  (Production: 10k-50k configs, 50k-100k spins each)")

"""
    simulate_one_config(params_phys, sequences; n_spins, rng_seed)

Run Monte Carlo simulation for one microstructure configuration.

Returns a Float32 vector of length N_MEAS with normalised signal values.
Each element is transverse(readout) / n_spins, giving attenuation in [0, 1].
"""
function simulate_one_config(
    params_phys::AbstractVector,
    sequences::Vector{Sequence};
    n_spins::Int = N_SPINS,
    rng_seed::Int = 0,
)
    r, f, d_par, d_extra = params_phys

    # Build packed-cylinder geometry.
    # Target volume fraction f is achieved by setting the cylinder repeats
    # so that pi*r^2 / (repeat_x * repeat_y) = f.
    # For a single cylinder centered at origin:
    #   repeat = r * sqrt(pi / f)
    repeat_dist = r * sqrt(pi / f)

    geometry = Cylinders(radius=r, repeats=[repeat_dist, repeat_dist])

    # Intra-axonal diffusivity is the parallel diffusivity.
    # Extra-axonal diffusivity is lower. We use a volume-weighted average
    # for the global diffusivity parameter, which MCMRSimulator uses for
    # all spins. The signal difference between compartments arises from
    # the geometry restricting diffusion perpendicular to cylinders.
    #
    # For a more accurate two-compartment model, one would set
    # different diffusivities inside vs outside. MCMRSimulator does not
    # directly support per-compartment diffusivity, so we use a
    # volume-weighted effective diffusivity. The restriction effect
    # from the cylinder walls is the primary contrast mechanism.
    d_eff = f * d_par + (1 - f) * d_extra

    # Create simulation with all sequences at once
    sim = Simulation(
        sequences;
        geometry    = geometry,
        diffusivity = d_eff,
        R2          = 0.0,   # ignore T2 decay for microstructure estimation
        R1          = 0.0,   # ignore T1 recovery
        verbose     = false, # suppress per-config printout
    )

    # Run Monte Carlo simulation
    # readout() with an integer creates that many random spins and
    # returns a SpinOrientationSum for each sequence.
    result = readout(n_spins, sim)

    # Extract transverse magnetisation (signal) for each sequence
    # result is a Vector{SpinOrientationSum} of length N_MEAS
    signals = Float32[transverse(result[i]) / n_spins for i in 1:length(sequences)]

    return signals
end

# Generate full training dataset
t_start = time()

train_theta_norm = sample_prior(rng, N_TRAIN)
train_theta_phys = denormalise(train_theta_norm)
train_signals    = zeros(Float32, N_MEAS, N_TRAIN)

for j in 1:N_TRAIN
    train_signals[:, j] = simulate_one_config(
        train_theta_phys[:, j], sequences;
        n_spins = N_SPINS,
        rng_seed = j,
    )

    if j % max(1, N_TRAIN ÷ 10) == 0 || j == 1
        elapsed = time() - t_start
        rate = j / elapsed
        eta = (N_TRAIN - j) / rate
        @printf("  [MCMR] %d/%d configs  (%.1f configs/s, ETA %.0fs)\n",
                j, N_TRAIN, rate, eta)
    end
end

t_mcmr = time() - t_start
@printf("  MCMRSimulator data generation: %.1fs (%.2f s/config)\n",
        t_mcmr, t_mcmr / N_TRAIN)

# Add Rician noise and b0-normalise
println("  Adding Rician noise (SNR 15-40) and b0-normalising...")
noisy_signals = copy(train_signals)
for j in 1:N_TRAIN
    snr = rand(rng) * 25.0 + 15.0  # SNR in [15, 40]
    sigma = 1.0 / snr
    n1 = randn(rng, Float32, N_MEAS) .* Float32(sigma)
    n2 = randn(rng, Float32, N_MEAS) .* Float32(sigma)
    noisy_signals[:, j] = @. sqrt((train_signals[:, j] + n1)^2 + n2^2)
end

# b0-normalise: divide by mean of b=0 measurements
for j in 1:N_TRAIN
    b0_mean = mean(noisy_signals[B0_MASK, j])
    b0_mean = max(b0_mean, 1f-6)
    noisy_signals[:, j] ./= b0_mean
end

println("  Signal statistics:")
@printf("    Clean range:  [%.4f, %.4f]\n", extrema(train_signals)...)
@printf("    Noisy+norm:   [%.4f, %.4f]\n", extrema(noisy_signals)...)

# ====================================================================
# 4. Train neural surrogate (optional speed-up)
# ====================================================================
println("\n--- Step 4: Training neural surrogate ---")
println("  This replaces expensive MCMR forward calls with a fast MLP.")
println("  ($SURROGATE_STEPS steps; production: 50k+)")

surrogate = build_surrogate(
    param_dim   = PARAM_DIM,
    signal_dim  = N_MEAS,
    hidden_dim  = 128,   # production: 256
    depth       = 4,     # production: 6
)
surr_ps, surr_st = Lux.setup(rng, surrogate)

# Data generator that samples from our pre-computed dataset
# (in production, one would generate fresh MCMR data on the fly)
function surrogate_data_fn(rng, n)
    idx = rand(rng, 1:N_TRAIN, n)
    params = train_theta_norm[:, idx]
    sigs   = noisy_signals[:, idx]
    return params, sigs
end

t_start = time()
surr_ps, surr_st, surr_losses = train_surrogate!(
    surrogate, surr_ps, surr_st, surrogate_data_fn;
    n_steps       = SURROGATE_STEPS,
    batch_size    = min(BATCH_SIZE, N_TRAIN),
    learning_rate = 1e-3,
    print_every   = max(1, SURROGATE_STEPS ÷ 5),
    loss_type     = :relative_mse,
)
t_surr = time() - t_start
@printf("  Surrogate training: %.1fs\n", t_surr)
@printf("  Final surrogate loss: %.6f\n", surr_losses[end])

# Validate surrogate accuracy
val_idx  = rand(rng, 1:N_TRAIN, 50)
val_pred, _ = surrogate(train_theta_norm[:, val_idx], surr_ps, surr_st)
val_true = noisy_signals[:, val_idx]
rel_err  = mean(abs.(val_pred .- val_true) ./ max.(abs.(val_true), 0.01f0))
@printf("  Surrogate validation rel. error: %.4f (%.1f%%)\n", rel_err, rel_err * 100)

# ====================================================================
# 5. Train conditional score network
# ====================================================================
println("\n--- Step 5: Training score network ---")
println("  Denoising score matching with VP-SDE schedule")
println("  ($SCORE_STEPS steps; production: 30k-100k)")

schedule = VPSchedule()

# Build score network
score_model = build_score_net(
    param_dim  = PARAM_DIM,
    signal_dim = N_MEAS,
    hidden_dim = SCORE_HIDDEN,
    depth      = SCORE_DEPTH,
    cond_dim   = 64,     # production: 128
)

# ScoreNetwork is a proper Lux model -- single setup call handles everything
score_ps, score_st = Lux.setup(rng, score_model)

# Fast surrogate-based data generator for score training
function score_sim_fn(rng, theta_norm)
    n = size(theta_norm, 2)
    signals_clean, _ = surrogate(theta_norm, surr_ps, surr_st)

    # Add Rician noise with variable SNR
    snr = rand(rng, Float32, 1, n) .* 25f0 .+ 15f0
    sigma = 1f0 ./ snr
    n1 = randn(rng, Float32, size(signals_clean)) .* sigma
    n2 = randn(rng, Float32, size(signals_clean)) .* sigma
    noisy = @. sqrt((signals_clean + n1)^2 + n2^2)

    # b0-normalise
    b0_mean = mean(noisy[B0_MASK, :], dims=1)
    b0_mean = max.(b0_mean, 1f-6)
    return noisy ./ b0_mean
end

# Train the score network
t_start = time()
score_ps, score_st, score_losses = train_score!(
    score_model, score_ps, score_st;
    simulator_fn  = score_sim_fn,
    prior_fn      = sample_prior,
    schedule      = schedule,
    num_steps     = SCORE_STEPS,
    batch_size    = BATCH_SIZE,
    learning_rate = 3e-4,
    print_every   = max(1, SCORE_STEPS ÷ 5),
    prediction    = :eps,
)
t_score = time() - t_start
@printf("  Score training: %.1fs (%.0f steps/s)\n", t_score, SCORE_STEPS / t_score)

# ====================================================================
# 6. Posterior sampling for a test signal
# ====================================================================
println("\n--- Step 6: Posterior sampling ---")

# Generate a test signal from known ground-truth parameters
gt_params_norm = Float32[0.5, 0.6, 0.5, 0.5]  # normalised
gt_params_phys = denormalise(gt_params_norm)
@printf("  Ground truth: r=%.2f um, f=%.2f, d_par=%.2f, d_extra=%.2f\n",
        gt_params_phys...)

# Generate test signal via MCMRSimulator (not the surrogate!)
println("  Running MCMR simulation for test signal (10k spins)...")
test_signal_clean = simulate_one_config(gt_params_phys, sequences; n_spins=10_000)

# Add noise and normalise
snr_test = 30.0
sigma_test = Float32(1.0 / snr_test)
n1 = randn(rng, Float32, N_MEAS) .* sigma_test
n2 = randn(rng, Float32, N_MEAS) .* sigma_test
test_signal = @. sqrt((test_signal_clean + n1)^2 + n2^2)
b0_mean_test = mean(test_signal[B0_MASK])
test_signal ./= max(b0_mean_test, 1f-6)

@printf("  Test signal range: [%.4f, %.4f]\n", extrema(test_signal)...)

# Draw posterior samples using DDPM sampler
println("  Drawing 200 posterior samples (500 diffusion steps)...")
# Production: 1000-5000 samples, 500-1000 steps
n_posterior = 200
n_diffusion_steps = 500

t_start = time()
posterior_samples = sample_posterior(
    score_model, score_ps, score_st, test_signal;
    schedule   = schedule,
    n_samples  = n_posterior,
    n_steps    = n_diffusion_steps,
    n_scalars  = PARAM_DIM,
    n_vectors  = 0,          # no orientation vectors in this model
    prediction = :eps,
)
t_sample = time() - t_start
@printf("  Posterior sampling: %.1fs (%.0f samples/s)\n",
        t_sample, n_posterior / t_sample)

# Convert posterior to physical units
posterior_phys = denormalise(posterior_samples)  # (PARAM_DIM, n_posterior)

# Clamp to prior bounds (samples may slightly exceed [0,1])
posterior_phys = clamp.(posterior_phys, LOWS, HIGHS)

println("  Posterior summary (mean +/- std):")
for (i, name) in enumerate(PARAM_NAMES)
    m = mean(posterior_phys[i, :])
    s = std(posterior_phys[i, :])
    gt = gt_params_phys[i]
    @printf("    %-8s  GT=%.3f  posterior=%.3f +/- %.3f\n", name, gt, m, s)
end

# ====================================================================
# 7. Visualisation
# ====================================================================
println("\n--- Step 7: Visualisation ---")

fig = Figure(size=(1200, 800))

# --- Row 1: Training diagnostics ---
ax1 = Axis(fig[1, 1]; xlabel="Step", ylabel="Loss",
           title="Surrogate Training Loss", yscale=log10)
lines!(ax1, 1:length(surr_losses), surr_losses; color=:blue)

ax2 = Axis(fig[1, 2]; xlabel="Step", ylabel="Loss",
           title="Score Network Training Loss", yscale=log10)
lines!(ax2, 1:length(score_losses), score_losses; color=:red)

# --- Row 2: Posterior distributions ---
param_labels = ["r (um)", "f (vol. frac.)", "d_par (um^2/ms)", "d_extra (um^2/ms)"]
param_colors = [:dodgerblue, :coral, :mediumseagreen, :mediumpurple]

for (i, (label, col)) in enumerate(zip(param_labels, param_colors))
    row = 2 + (i - 1) ÷ 2
    col_idx = ((i - 1) % 2) + 1
    ax = Axis(fig[row, col_idx]; xlabel=label, ylabel="Density",
              title="Posterior: $label")

    hist!(ax, posterior_phys[i, :]; bins=25, color=(col, 0.6),
          strokecolor=:black, strokewidth=0.5, normalization=:pdf)
    vlines!(ax, [gt_params_phys[i]]; color=:red, linewidth=2.5,
            linestyle=:dash, label="Ground truth")
    vlines!(ax, [mean(posterior_phys[i, :])]; color=:black, linewidth=1.5,
            linestyle=:dot, label="Posterior mean")
    axislegend(ax; position=:rt, labelsize=10)
end

save(joinpath(@__DIR__, "mcmr_restricted_diffusion.png"), fig; px_per_unit=2)
println("  Saved figure: examples/mcmr_restricted_diffusion.png")

# ====================================================================
# Summary
# ====================================================================
println("\n" * "=" ^ 70)
println("PIPELINE COMPLETE")
println("=" ^ 70)
@printf("  MCMR data generation : %6.1fs  (%d configs, %d spins each)\n",
        t_mcmr, N_TRAIN, N_SPINS)
@printf("  Surrogate training   : %6.1fs  (%d steps)\n", t_surr, SURROGATE_STEPS)
@printf("  Score training       : %6.1fs  (%d steps)\n", t_score, SCORE_STEPS)
@printf("  Posterior sampling   : %6.1fs  (%d samples)\n", t_sample, n_posterior)
println()
println("Production recommendations:")
println("  - N_TRAIN:       10,000 - 50,000  (currently $N_TRAIN)")
println("  - N_SPINS:       50,000 - 100,000 (currently $N_SPINS)")
println("  - SURROGATE_STEPS: 50,000+         (currently $SURROGATE_STEPS)")
println("  - SCORE_STEPS:   30,000 - 100,000 (currently $SCORE_STEPS)")
println("  - BATCH_SIZE:    256 - 512         (currently $BATCH_SIZE)")
println("  - SCORE_HIDDEN:  512               (currently $SCORE_HIDDEN)")
println("  - n_posterior:   1,000 - 5,000     (currently $n_posterior)")
println("  - Add orientation dispersion (Watson/Bingham) for realistic WM")
println("  - Use per-compartment diffusivity once MCMRSimulator supports it")
println("  - Consider CUDA acceleration for score training (LuxCUDA)")
