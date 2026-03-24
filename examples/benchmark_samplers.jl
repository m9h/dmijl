#!/usr/bin/env julia
"""
Benchmark: DDPM (hand-rolled) vs DiffEq EM vs DiffEq adaptive ODE.

Uses a dummy score function to measure pure solver overhead.
"""

const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "src/diffusion/schedule.jl"))
include(joinpath(ROOT, "src/diffusion/sample_diffeq.jl"))

using Random, Statistics, LinearAlgebra, Printf

# Dummy score function: returns zeros (pure overhead test)
dummy_score(theta, t, signal) = zeros(Float32, length(theta))

signal = randn(Float32, 90)
schedule = VPSchedule()
n_samples = 50

println("=" ^ 60)
println("SAMPLER BENCHMARK (dummy score, $n_samples samples)")
println("=" ^ 60)

# Warmup
sample_posterior_diffeq(dummy_score, signal, schedule;
    n_samples=2, solver=EM(), dt=0.01)
sample_posterior_ode(dummy_score, signal, schedule;
    n_samples=2, solver=Tsit5(), dt=0.01)

# Benchmark DiffEq EM (fixed step)
for dt in [0.01, 0.005, 0.002]
    steps = round(Int, 1.0 / dt)
    t = @elapsed samples = sample_posterior_diffeq(
        dummy_score, signal, schedule;
        n_samples=n_samples, solver=EM(), dt=dt,
    )
    @printf("  DiffEq EM  (dt=%.3f, %d steps): %.2fs  (%.0f samples/s)\n",
            dt, steps, t, n_samples/t)
end

# Benchmark DiffEq ODE (adaptive — Tsit5)
t = @elapsed samples = sample_posterior_ode(
    dummy_score, signal, schedule;
    n_samples=n_samples, solver=Tsit5(),
)
@printf("  DiffEq ODE (Tsit5 adaptive):    %.2fs  (%.0f samples/s)\n",
        t, n_samples/t)

# Benchmark DiffEq ODE (fixed step — Euler)
# Euler ODE needs dt passed to solve(), skip for now
# The adaptive Tsit5 solver above is the recommended ODE approach

println("\nNote: with a trained score network, the score evaluation")
println("will dominate runtime. This measures solver overhead only.")
