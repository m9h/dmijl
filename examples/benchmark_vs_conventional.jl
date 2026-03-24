#!/usr/bin/env julia
"""
Benchmark: Score-based posterior estimation vs conventional microstructure fitting.

Compares three methods on identical synthetic Ball+2Stick data:
  1. Nonlinear least squares (Optim.jl, per-voxel) -- skipped if Optim not installed
  2. Grid search MAP (coarse likelihood grid)
  3. Score-based posterior (ours): train score network, sample posterior

Metrics: RMSE, angular error, Pearson r, wall-clock time, calibration.

Demo-scale settings below. For production:
  - N_TEST = 5000+, N_TRAIN_STEPS = 100_000, GRID_PTS = 15+
  - hidden_dim = 512, depth = 8, n_posterior_samples = 2000
  - Run on GPU via LuxCUDA
"""

const ROOT = joinpath(@__DIR__, "..")

using Pkg
Pkg.activate(ROOT)

using Lux, Random, Statistics, LinearAlgebra, Printf, ComponentArrays

include(joinpath(ROOT, "src/models/ball_stick.jl"))
include(joinpath(ROOT, "src/noise.jl"))
include(joinpath(ROOT, "src/diffusion/schedule.jl"))
include(joinpath(ROOT, "src/diffusion/score_net.jl"))
include(joinpath(ROOT, "src/diffusion/train.jl"))
include(joinpath(ROOT, "src/diffusion/sample.jl"))
include(joinpath(ROOT, "src/validation/metrics.jl"))

# ---------------------------------------------------------------------------
# Conditional Optim.jl loading (not in Project.toml)
# ---------------------------------------------------------------------------
const HAS_OPTIM = Ref(false)
try
    @eval using Optim
    HAS_OPTIM[] = true
    println("[deps] Optim.jl loaded -- NLS fitting enabled")
catch
    @warn "Optim.jl not available; NLS benchmark will be skipped. " *
          "Install with: using Pkg; Pkg.add(\"Optim\")"
end

# ---------------------------------------------------------------------------
# Configuration -- demo scale (see docstring for production values)
# ---------------------------------------------------------------------------
const N_TEST           = 200      # production: 1000-5000
const N_TRAIN_STEPS    = 5_000    # production: 50_000-100_000
const BATCH_SIZE       = 256      # production: 512
const SNR              = 20.0
const GRID_PTS         = 5        # production: 10-15 per scalar dim
const N_POSTERIOR      = 100      # production: 500-2000
const N_POSTERIOR_STEPS = 100     # production: 500
const HIDDEN_DIM       = 128      # production: 512
const DEPTH            = 4        # production: 6-8

# ---------------------------------------------------------------------------
# HCP-like acquisition
# ---------------------------------------------------------------------------
rng = Random.default_rng()
Random.seed!(rng, 42)

n_b0, n_b1, n_b2, n_b3 = 6, 30, 30, 24
bvals = vcat(zeros(n_b0), fill(1e9, n_b1), fill(2e9, n_b2), fill(3e9, n_b3))
n_meas = length(bvals)

function rand_unit_vecs(rng, n)
    z = randn(rng, n, 3)
    return z ./ sqrt.(sum(z.^2, dims=2))
end

bvecs = vcat(
    repeat([1.0 0.0 0.0], n_b0, 1),
    rand_unit_vecs(rng, n_b1),
    rand_unit_vecs(rng, n_b2),
    rand_unit_vecs(rng, n_b3),
)

model_phys = BallStickModel(bvals, bvecs)
b0_mask = bvals .< 100e6

# ---------------------------------------------------------------------------
# Parameter ranges (physical units)
# Ball+2Stick: [d_ball, d_stick, f1, f2, mu1x, mu1y, mu1z, mu2x, mu2y, mu2z]
# ---------------------------------------------------------------------------
const PARAM_NAMES = ["d_ball", "d_stick", "f1", "f2",
                     "mu1x", "mu1y", "mu1z", "mu2x", "mu2y", "mu2z"]

const LOWS  = Float32[1.0e-9, 0.5e-9, 0.1,  0.05, -1, -1, 0, -1, -1, 0]
const HIGHS = Float32[3.5e-9, 2.5e-9, 0.8,  0.5,   1,  1, 1,  1,  1, 1]
const SPANS = max.(HIGHS .- LOWS, 1f-12)

# ---------------------------------------------------------------------------
# Helper: physical <-> normalised [0,1] parameter space
# ---------------------------------------------------------------------------
to_norm(theta_phys)   = (theta_phys .- LOWS) ./ SPANS
from_norm(theta_norm) = LOWS .+ theta_norm .* SPANS

# ---------------------------------------------------------------------------
# Prior sampling (normalised), with f1 >= f2 label-ordering
# ---------------------------------------------------------------------------
function sample_prior(rng, n)
    theta_norm = rand(rng, Float32, 10, n)
    theta = from_norm(theta_norm)
    # Enforce f1 >= f2 and swap corresponding orientations
    for j in 1:n
        if theta[3, j] < theta[4, j]
            theta[3, j], theta[4, j] = theta[4, j], theta[3, j]
            theta[5:7, j], theta[8:10, j] = theta[8:10, j], theta[5:7, j]
        end
        # Normalise orientations to unit sphere
        for s in (5, 8)
            v = @view theta[s:s+2, j]
            v ./= max(norm(v), 1e-8)
        end
    end
    return to_norm(theta)
end

# ---------------------------------------------------------------------------
# Simulator function for training: prior -> forward model + noise + b0 norm
# ---------------------------------------------------------------------------
function sim_fn(rng, theta_norm)
    n = size(theta_norm, 2)
    theta = from_norm(theta_norm)
    signals = zeros(Float32, n_meas, n)
    for j in 1:n
        signals[:, j] = simulate(model_phys, @view theta[:, j])
    end
    # Rician noise with variable SNR (training sees a range)
    noisy = add_rician_noise(rng, signals'; snr_range=(10.0, 50.0))'
    b0_mean = max.(mean(noisy[b0_mask, :], dims=1), 1f-6)
    return noisy ./ b0_mean
end

# ---------------------------------------------------------------------------
# Generate ground-truth test data at fixed SNR
# ---------------------------------------------------------------------------
println("=" ^ 70)
println("BENCHMARK: Score Posterior vs Conventional Fitting")
println("  N_TEST=$N_TEST  SNR=$SNR  GRID_PTS=$GRID_PTS")
println("  Score: steps=$N_TRAIN_STEPS  hidden=$HIDDEN_DIM  depth=$DEPTH")
println("=" ^ 70)

println("\n[1/6] Generating ground truth...")
theta_test_norm = sample_prior(rng, N_TEST)
theta_test_phys = from_norm(theta_test_norm)

# Forward model (clean)
signals_clean = zeros(Float32, n_meas, N_TEST)
for j in 1:N_TEST
    signals_clean[:, j] = simulate(model_phys, @view theta_test_phys[:, j])
end

# Add Rician noise at fixed SNR
sigma_noise = Float32(1.0 / SNR)
n1 = randn(rng, Float32, size(signals_clean)) .* sigma_noise
n2 = randn(rng, Float32, size(signals_clean)) .* sigma_noise
signals_noisy = @. sqrt((signals_clean + n1)^2 + n2^2)

# b0-normalise
b0_mean = max.(mean(signals_noisy[b0_mask, :], dims=1), 1f-6)
signals_test = signals_noisy ./ b0_mean

println("  theta shape: $(size(theta_test_phys))  signal shape: $(size(signals_test))")
println("  Signal range: $(round.(extrema(signals_test), digits=3))")

# ═══════════════════════════════════════════════════════════════════════════
# METHOD 1: Nonlinear Least Squares (Optim.jl) -- per-voxel
# ═══════════════════════════════════════════════════════════════════════════

theta_nls = zeros(Float32, 10, N_TEST)
time_nls = NaN
nls_succeeded = false

if HAS_OPTIM[]
    println("\n[2/6] Method 1 -- Nonlinear Least Squares (NelderMead, per-voxel)...")

    # Bounds for NelderMead initialisation (physical units)
    lower_bounds = Float64.(LOWS)
    upper_bounds = Float64.(HIGHS)

    function nls_objective(params, signal_obs)
        pred = simulate(model_phys, params)
        # b0-normalise prediction
        b0_pred = max(mean(pred[b0_mask]), 1e-8)
        pred_norm = pred ./ b0_pred
        return sum((pred_norm .- signal_obs).^2)
    end

    t_nls_start = time()
    n_nls_fail = 0

    for j in 1:N_TEST
        sig_obs = Float64.(signals_test[:, j])

        # Random initial guess within bounds (3 restarts, keep best)
        best_loss = Inf
        best_params = nothing

        for restart in 1:3
            x0 = lower_bounds .+ rand(rng, 10) .* (upper_bounds .- lower_bounds)
            # Normalise orientation vectors
            x0[5:7] ./= max(norm(x0[5:7]), 1e-8)
            x0[8:10] ./= max(norm(x0[8:10]), 1e-8)

            try
                result = @eval Optim.optimize(
                    p -> $nls_objective(p, $sig_obs),
                    $lower_bounds, $upper_bounds, $x0,
                    Optim.Fminbox(Optim.NelderMead()),
                    Optim.Options(iterations=2000, f_tol=1e-8),
                )
                if result.minimum < best_loss
                    best_loss = result.minimum
                    best_params = result.minimizer
                end
            catch e
                n_nls_fail += 1
            end
        end

        if best_params !== nothing
            theta_nls[:, j] = Float32.(best_params)
        else
            # Fallback: midpoint of bounds
            theta_nls[:, j] = Float32.((lower_bounds .+ upper_bounds) ./ 2)
        end

        if j % 50 == 0
            @printf("    NLS: %d/%d voxels (%.1fs)\n", j, N_TEST, time() - t_nls_start)
        end
    end

    time_nls = time() - t_nls_start
    nls_succeeded = true
    @printf("  NLS completed: %.1fs total (%.1f ms/voxel)\n",
            time_nls, 1000 * time_nls / N_TEST)
    if n_nls_fail > 0
        println("  NLS failures: $n_nls_fail / $(N_TEST * 3) optimisations")
    end
else
    println("\n[2/6] Method 1 -- NLS skipped (Optim.jl not available)")
end

# ═══════════════════════════════════════════════════════════════════════════
# METHOD 2: Grid Search MAP -- coarse exhaustive search
# ═══════════════════════════════════════════════════════════════════════════

println("\n[3/6] Method 2 -- Grid Search MAP ($GRID_PTS pts per scalar dim)...")

# Build parameter grid for scalar params only; orientations are sampled
# from a discrete set of unit vectors on the sphere
function fibonacci_sphere(n)
    # Fibonacci lattice on hemisphere (z >= 0)
    pts = zeros(Float64, n, 3)
    golden = (1 + sqrt(5)) / 2
    for i in 1:n
        theta = acos(1 - (2 * (i - 0.5)) / (2 * n))  # hemisphere
        phi = 2 * pi * (i - 1) / golden
        pts[i, :] = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
    end
    return pts
end

n_orient = max(GRID_PTS, 6)  # number of discrete orientations per fiber
orient_grid = fibonacci_sphere(n_orient)

# Scalar grid: d_ball, d_stick, f1, f2
scalar_grids = [
    range(LOWS[i], HIGHS[i], length=GRID_PTS) for i in 1:4
]

# Pre-compute all candidate parameter vectors
# Full grid would be GRID_PTS^4 * n_orient^2 -- too large.
# Strategy: coarse scalar grid x small orient grid
n_grid_total = GRID_PTS^4 * n_orient^2
println("  Grid size: $(GRID_PTS)^4 * $(n_orient)^2 = $n_grid_total candidates")

# For demo, if grid is too large truncate orient grid
if n_grid_total > 500_000
    n_orient = max(4, round(Int, sqrt(500_000 / GRID_PTS^4)))
    orient_grid = fibonacci_sphere(n_orient)
    n_grid_total = GRID_PTS^4 * n_orient^2
    println("  Reduced orient grid to $n_orient pts ($n_grid_total candidates)")
end

# Build grid matrix (10, n_grid_total)
grid_params = zeros(Float64, 10, n_grid_total)
idx = 0
for d_ball in scalar_grids[1], d_stick in scalar_grids[2],
    f1 in scalar_grids[3], f2 in scalar_grids[4]
    # Skip invalid: f1 + f2 > 1 or f1 < f2
    (f1 + f2 > 0.95 || f1 < f2) && continue
    for oi1 in 1:n_orient, oi2 in 1:n_orient
        oi1 == oi2 && continue  # distinct fibers
        idx += 1
        if idx > n_grid_total
            break
        end
        grid_params[1, idx] = d_ball
        grid_params[2, idx] = d_stick
        grid_params[3, idx] = f1
        grid_params[4, idx] = f2
        grid_params[5:7, idx] = orient_grid[oi1, :]
        grid_params[8:10, idx] = orient_grid[oi2, :]
    end
end
n_valid = idx
grid_params = grid_params[:, 1:n_valid]
println("  Valid grid points: $n_valid (after constraints)")

# Pre-simulate signals for the entire grid
println("  Pre-computing grid signals...")
grid_signals = zeros(Float64, n_meas, n_valid)
for j in 1:n_valid
    grid_signals[:, j] = simulate(model_phys, @view grid_params[:, j])
end
# b0-normalise grid signals
grid_b0 = max.(mean(grid_signals[b0_mask, :], dims=1), 1e-8)
grid_signals ./= grid_b0

# Grid search: for each test voxel, find best-matching grid point (L2 on signals)
theta_grid = zeros(Float32, 10, N_TEST)
t_grid_start = time()

for j in 1:N_TEST
    sig_obs = Float64.(signals_test[:, j])
    # Vectorised L2 distance
    residuals = grid_signals .- sig_obs
    sse = vec(sum(residuals.^2, dims=1))
    best_idx = argmin(sse)
    theta_grid[:, j] = Float32.(grid_params[:, best_idx])

    if j % 100 == 0
        @printf("    Grid: %d/%d voxels (%.1fs)\n", j, N_TEST, time() - t_grid_start)
    end
end

time_grid = time() - t_grid_start
@printf("  Grid search completed: %.1fs total (%.1f ms/voxel)\n",
        time_grid, 1000 * time_grid / N_TEST)

# ═══════════════════════════════════════════════════════════════════════════
# METHOD 3: Score-based Posterior (ours)
# ═══════════════════════════════════════════════════════════════════════════

println("\n[4/6] Method 3 -- Score-based Posterior Estimation...")

# Build score network (demo-scale)
println("  Building FiLM-conditioned ScoreNetwork (hidden=$HIDDEN_DIM, depth=$DEPTH)...")
score_model = build_score_net(
    param_dim=10, signal_dim=n_meas,
    hidden_dim=HIDDEN_DIM, depth=DEPTH, cond_dim=64,
)
ps, st = Lux.setup(rng, score_model)
ps = ComponentArray(ps)  # For Optimisers.jl

# Train
println("  Training score network ($N_TRAIN_STEPS steps, batch=$BATCH_SIZE)...")
ps, st, losses = train_score!(
    score_model, ps, st;
    simulator_fn=sim_fn,
    prior_fn=sample_prior,
    schedule=VPSchedule(),
    num_steps=N_TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    learning_rate=3e-4,
    print_every=max(N_TRAIN_STEPS ÷ 5, 1),
)
@printf("  Final training loss: %.4f\n", losses[end])

# Sample posteriors for all test signals
println("  Sampling posteriors for $N_TEST test voxels...")
theta_score = zeros(Float32, 10, N_TEST)
theta_score_std = zeros(Float32, 10, N_TEST)
st_eval = Lux.testmode(st)

t_score_start = time()
for j in 1:N_TEST
    sig_vec = Float32.(signals_test[:, j])  # (n_meas,) vector

    samples = sample_posterior(
        score_model, ps, st_eval, sig_vec;
        schedule=VPSchedule(),
        n_samples=N_POSTERIOR,
        n_steps=N_POSTERIOR_STEPS,
        n_scalars=4,
        n_vectors=2,
    )
    # samples: (10, N_POSTERIOR) in normalised space
    # Posterior mean as point estimate
    theta_score[:, j] = from_norm(mean(samples, dims=2)[:, 1])
    # Posterior std as uncertainty (in physical units)
    theta_score_std[:, j] = SPANS .* vec(std(samples, dims=2))

    if j % 50 == 0
        @printf("    Score: %d/%d voxels (%.1fs)\n", j, N_TEST, time() - t_score_start)
    end
end

time_score = time() - t_score_start
@printf("  Score sampling completed: %.1fs total (%.1f ms/voxel)\n",
        time_score, 1000 * time_score / N_TEST)

# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

println("\n[5/6] Evaluating all methods...")

scalar_names = ["d_ball", "d_stick", "f1", "f2"]

function evaluate_method(name, theta_est)
    println("\n  --- $name ---")

    # Scalar RMSE and Pearson r
    for (i, pname) in enumerate(scalar_names)
        r = pearson_r(vec(theta_test_phys[i, :]), vec(theta_est[i, :]))
        err = rmse(vec(theta_est[i, :]), vec(theta_test_phys[i, :]))
        @printf("    %-8s  RMSE=%.4e  r=%.3f\n", pname, err, r)
    end

    # Angular errors with label-switching
    result = evaluate_ball2stick(theta_test_phys, theta_est; n_scalars=4)
    @printf("    fiber1   median=%.1f deg  mean=%.1f deg\n",
            result.fiber1_median, result.fiber1_mean)
    @printf("    fiber2   median=%.1f deg  mean=%.1f deg\n",
            result.fiber2_median, result.fiber2_mean)

    return result
end

results = Dict{String, Any}()

if nls_succeeded
    results["NLS"] = evaluate_method("Nonlinear Least Squares", theta_nls)
end
results["Grid"] = evaluate_method("Grid Search MAP", theta_grid)
results["Score"] = evaluate_method("Score Posterior (ours)", theta_score)

# ---------------------------------------------------------------------------
# Calibration: does posterior uncertainty correlate with actual error?
# (Score method only)
# ---------------------------------------------------------------------------
println("\n  --- Calibration (Score Posterior) ---")
println("  Does posterior std correlate with |estimation error|?")

for (i, pname) in enumerate(scalar_names)
    abs_err = abs.(theta_score[i, :] .- theta_test_phys[i, :])
    unc = theta_score_std[i, :]
    cal_r = pearson_r(vec(unc), vec(abs_err))
    @printf("    %-8s  uncertainty-error r=%.3f\n", pname, cal_r)
end

# Angular calibration: use mean orientation uncertainty as proxy
orient_unc_1 = vec(sqrt.(sum(theta_score_std[5:7, :].^2, dims=1)))
orient_unc_2 = vec(sqrt.(sum(theta_score_std[8:10, :].^2, dims=1)))
orient_err_1 = results["Score"].errors1
orient_err_2 = results["Score"].errors2

cal_mu1 = pearson_r(orient_unc_1, orient_err_1)
cal_mu2 = pearson_r(orient_unc_2, orient_err_2)
@printf("    fiber1   uncertainty-error r=%.3f\n", cal_mu1)
@printf("    fiber2   uncertainty-error r=%.3f\n", cal_mu2)

# ---------------------------------------------------------------------------
# Timing summary
# ---------------------------------------------------------------------------
println("\n  --- Wall-Clock Timing ---")
if nls_succeeded
    @printf("    NLS:    %7.1f ms/voxel  (total %.1fs)\n",
            1000 * time_nls / N_TEST, time_nls)
end
@printf("    Grid:   %7.1f ms/voxel  (total %.1fs)\n",
        1000 * time_grid / N_TEST, time_grid)
@printf("    Score:  %7.1f ms/voxel  (total %.1fs, excl. training)\n",
        1000 * time_score / N_TEST, time_score)
println("    Note: Score training took $N_TRAIN_STEPS steps -- amortised over all voxels.")

# ═══════════════════════════════════════════════════════════════════════════
# VISUALISATION (CairoMakie)
# ═══════════════════════════════════════════════════════════════════════════

println("\n[6/6] Generating plots with CairoMakie...")

using CairoMakie

# Collect methods for plotting
method_names = String[]
method_estimates = Dict{String, Matrix{Float32}}()
method_colors = Dict{String, Symbol}()
method_times = Dict{String, Float64}()

if nls_succeeded
    push!(method_names, "NLS")
    method_estimates["NLS"] = theta_nls
    method_colors["NLS"] = :dodgerblue
    method_times["NLS"] = time_nls
end
push!(method_names, "Grid")
method_estimates["Grid"] = theta_grid
method_colors["Grid"] = :orange
method_times["Grid"] = time_grid

push!(method_names, "Score")
method_estimates["Score"] = theta_score
method_colors["Score"] = :crimson
method_times["Score"] = time_score

n_methods = length(method_names)

# ---- Figure 1: Estimated vs True scatter (scalar params) ----
fig1 = Figure(size=(400 * n_methods, 350 * 4), fontsize=12)
for (mi, mname) in enumerate(method_names)
    est = method_estimates[mname]
    for (pi, pname) in enumerate(scalar_names)
        ax = Axis(fig1[pi, mi],
            xlabel = pi == 4 ? "True $pname" : "",
            ylabel = mi == 1 ? "Estimated $pname" : "",
            title  = pi == 1 ? mname : "",
        )
        true_vals = vec(theta_test_phys[pi, :])
        est_vals  = vec(est[pi, :])
        scatter!(ax, true_vals, est_vals, markersize=3, alpha=0.4,
                 color=method_colors[mname])
        lo, hi = extrema(vcat(true_vals, est_vals))
        lines!(ax, [lo, hi], [lo, hi], color=:black, linestyle=:dash, linewidth=1)

        r = pearson_r(true_vals, est_vals)
        text!(ax, lo + 0.05*(hi-lo), hi - 0.1*(hi-lo),
              text="r=$(round(r, digits=3))", fontsize=10)
    end
end
Label(fig1[0, :], "Estimated vs True: Scalar Parameters", fontsize=16, font=:bold)

outdir = joinpath(ROOT, "examples", "benchmark_outputs")
mkpath(outdir)
save(joinpath(outdir, "scatter_scalars.png"), fig1, px_per_unit=2)
println("  Saved scatter_scalars.png")

# ---- Figure 2: Angular error histograms ----
fig2 = Figure(size=(400 * n_methods, 350 * 2), fontsize=12)
for (mi, mname) in enumerate(method_names)
    r = results[mname]
    for (fi, (errors, label)) in enumerate([(r.errors1, "Fiber 1"), (r.errors2, "Fiber 2")])
        ax = Axis(fig2[fi, mi],
            xlabel = fi == 2 ? "Angular error (deg)" : "",
            ylabel = mi == 1 ? "Count" : "",
            title  = fi == 1 ? "$mname" : "",
        )
        hist!(ax, errors, bins=30, color=(method_colors[mname], 0.6))
        med = median(errors)
        vlines!(ax, [med], color=:black, linestyle=:dash, linewidth=1.5)
        text!(ax, med + 1, 0, text="med=$(round(med, digits=1))", fontsize=9)
        Label(fig2[fi, 0], label, fontsize=11, rotation=pi/2)
    end
end
Label(fig2[0, :], "Angular Error Distributions", fontsize=16, font=:bold)
save(joinpath(outdir, "angular_errors.png"), fig2, px_per_unit=2)
println("  Saved angular_errors.png")

# ---- Figure 3: Timing bar chart ----
fig3 = Figure(size=(500, 350), fontsize=13)
ax3 = Axis(fig3[1, 1],
    ylabel="ms per voxel",
    title="Wall-Clock Time per Voxel (inference only)",
    xticks=(1:n_methods, method_names),
)
times_ms = [1000 * method_times[m] / N_TEST for m in method_names]
barplot!(ax3, 1:n_methods, times_ms,
         color=[method_colors[m] for m in method_names])
for (i, t) in enumerate(times_ms)
    text!(ax3, i, t + maximum(times_ms)*0.03,
          text="$(round(t, digits=1))", align=(:center, :bottom), fontsize=10)
end
save(joinpath(outdir, "timing_bar.png"), fig3, px_per_unit=2)
println("  Saved timing_bar.png")

# ---- Figure 4: Calibration plot (score posterior only) ----
fig4 = Figure(size=(400 * 2, 350 * 2), fontsize=12)
for (pi, pname) in enumerate(scalar_names)
    row, col = divrem(pi - 1, 2) .+ (1, 1)
    ax = Axis(fig4[row, col],
        xlabel="|Estimation error|",
        ylabel="Posterior std",
        title="Calibration: $pname",
    )
    abs_err = abs.(theta_score[pi, :] .- theta_test_phys[pi, :])
    unc = theta_score_std[pi, :]
    scatter!(ax, vec(abs_err), vec(unc), markersize=3, alpha=0.4, color=:crimson)
    r = pearson_r(vec(unc), vec(abs_err))
    lo, hi = 0.0, max(maximum(abs_err), maximum(unc)) * 1.1
    text!(ax, lo + 0.05*(hi-lo), hi - 0.1*(hi-lo),
          text="r=$(round(r, digits=3))", fontsize=10)
end
Label(fig4[0, :], "Score Posterior Calibration", fontsize=16, font=:bold)
save(joinpath(outdir, "calibration.png"), fig4, px_per_unit=2)
println("  Saved calibration.png")

# ---- Figure 5: Training loss curve ----
fig5 = Figure(size=(600, 300), fontsize=13)
ax5 = Axis(fig5[1, 1],
    xlabel="Training step",
    ylabel="MSE loss",
    title="Score Network Training Loss",
    yscale=log10,
)
lines!(ax5, 1:length(losses), losses, color=:crimson, linewidth=1)
save(joinpath(outdir, "training_loss.png"), fig5, px_per_unit=2)
println("  Saved training_loss.png")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

header = @sprintf("%-12s  %-10s  %-10s  %-10s  %-10s  %-8s  %-8s  %s",
    "Method", "RMSE_dball", "RMSE_f1", "r_f1", "r_f2",
    "fib1_med", "fib2_med", "ms/vox")
println(header)
println("-" ^ length(header))

for mname in method_names
    est = method_estimates[mname]
    r = results[mname]
    rmse_db = rmse(vec(est[1, :]), vec(theta_test_phys[1, :]))
    rmse_f1 = rmse(vec(est[3, :]), vec(theta_test_phys[3, :]))
    r_f1 = pearson_r(vec(theta_test_phys[3, :]), vec(est[3, :]))
    r_f2 = pearson_r(vec(theta_test_phys[4, :]), vec(est[4, :]))
    t_ms = 1000 * method_times[mname] / N_TEST

    @printf("%-12s  %.4e  %.4e  %10.3f  %10.3f  %6.1f   %6.1f   %7.1f\n",
        mname, rmse_db, rmse_f1, r_f1, r_f2,
        r.fiber1_median, r.fiber2_median, t_ms)
end

println("\nPlots saved to: $outdir")
println("Done.")
