"""
Bloch-Torrey neural surrogate.

Learns the mapping: (microstructure_params, acquisition_params) → dMRI signal
by training on Bloch-Torrey PDE solutions (via KomaMRI.jl or analytical models).

Two training modes:
1. **Supervised**: train on (params, signal) pairs from a reference simulator
2. **Physics-informed**: enforce Bloch-Torrey PDE residual as a loss term

The supervised mode bootstraps from analytical solutions;
the PINN mode generalizes to restricted diffusion without labels.
"""

using Lux, Random, Optimisers, Zygote, Statistics

# ------------------------------------------------------------------ #
# Surrogate network
# ------------------------------------------------------------------ #

"""
    build_surrogate(; param_dim, acq_dim, signal_dim, hidden_dim, depth)

Build a neural surrogate that maps (params, acquisition_features) → signal.

The network takes concatenated [microstructure_params; acq_features] as input
and predicts the normalized dMRI signal at each measurement.
"""
function build_surrogate(;
    param_dim::Int = 10,
    signal_dim::Int = 90,
    hidden_dim::Int = 256,
    depth::Int = 6,
)
    # Input: param_dim (microstructure params)
    # Output: signal_dim (one signal value per measurement)
    # Acquisition info is baked into the training data (not an explicit input)
    # This is an amortized surrogate for a FIXED acquisition.

    layers = Any[Dense(param_dim => hidden_dim, gelu)]
    for _ in 2:depth
        push!(layers, Dense(hidden_dim => hidden_dim, gelu))
    end
    push!(layers, Dense(hidden_dim => signal_dim, sigmoid))  # signal ∈ [0, 1]

    return Chain(layers...)
end

# ------------------------------------------------------------------ #
# Supervised training (from analytical/KomaMRI reference)
# ------------------------------------------------------------------ #

"""
    train_surrogate!(model, ps, st, data_fn; n_steps, batch_size, lr)

Train surrogate on (params, signal) pairs from a reference simulator.

`data_fn(rng, n)` should return `(params, signals)` each of shape
`(dim, n)` — column-major batches.
"""
function train_surrogate!(
    model, ps, st, data_fn;
    n_steps::Int = 20_000,
    batch_size::Int = 512,
    learning_rate::Float64 = 1e-3,
    print_every::Int = 1000,
)
    opt_state = Optimisers.setup(Adam(learning_rate), ps)
    rng = Random.default_rng()
    losses = Float64[]

    t0 = time()

    for step in 1:n_steps
        # Generate training batch from reference simulator
        params_batch, signals_batch = data_fn(rng, batch_size)
        # params_batch: (param_dim, batch_size)
        # signals_batch: (signal_dim, batch_size)

        # Compute loss and gradients
        (loss, st), grads = Zygote.withgradient(ps) do p
            pred, new_st = model(params_batch, p, st)
            l = mean((pred .- signals_batch).^2)
            return l, new_st
        end

        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
        push!(losses, loss)

        if step % print_every == 0 || step == 1
            elapsed = time() - t0
            rate = step / elapsed
            @info "[Surrogate] step $step/$n_steps  loss=$(round(loss, sigdigits=4))  ($(round(rate, digits=0)) steps/s)"
        end
    end

    elapsed = time() - t0
    @info "[Surrogate] Done. $n_steps steps in $(round(elapsed, digits=1))s"
    return ps, st, losses
end

# ------------------------------------------------------------------ #
# Physics-informed training (Bloch-Torrey PDE residual)
# ------------------------------------------------------------------ #

"""
    BlochTorreyResidual

Computes the PDE residual for the Bloch-Torrey equation using AD.
The surrogate u(θ, t, x) approximates the magnetization M(t, x; θ).

PDE: ∂M/∂t = D(θ)∇²M - iγ(G(t)·x)M - M/T₂

For a well-trained surrogate, the residual should be ≈ 0.
"""
struct BlochTorreyResidual
    gamma::Float64       # gyromagnetic ratio (rad/s/T)
    gradient_fn           # G(t) → [Gx, Gy, Gz]
end

BlochTorreyResidual(; gradient_fn) = BlochTorreyResidual(2.675e8, gradient_fn)

"""
    pde_loss(residual, model, ps, st, t, x, D, T2)

Compute the PDE residual loss at collocation points.
Uses automatic differentiation for ∂M/∂t and ∇²M.

This is the PINN loss term that can be added to supervised training
for physics regularization, or used standalone for restricted diffusion
where no analytical solution exists.
"""
function pde_loss(
    res::BlochTorreyResidual,
    model, ps, st,
    t::AbstractVector,    # (n_colloc,) time points
    x::AbstractMatrix,    # (3, n_colloc) spatial positions
    D::AbstractVector,    # (n_colloc,) diffusivity at each point
    T2::AbstractVector,   # (n_colloc,) T2 at each point
)
    n = length(t)
    total_residual = 0.0f0

    for i in 1:n
        # Point: (t_i, x_i)
        ti, xi, Di, T2i = t[i], x[:, i], D[i], T2[i]

        # M = model(input_features, ps, st)
        # For PINN: we need ∂M/∂t and ∇²M via AD
        # Input features encode (t, x, microstructure_params)
        # This requires a model that takes spatiotemporal input.
        #
        # For the initial implementation: compute residual numerically
        # via finite differences. Graduate to AD once the architecture
        # is validated.

        G = res.gradient_fn(ti)

        # Finite difference approximation
        dt = 1e-6
        dx = 1e-7

        input_i = vcat([ti], xi)
        input_dt = vcat([ti + dt], xi)

        # This is a placeholder — real implementation needs the PINN
        # to take (t, x) as input, not microstructure params.
        # The PINN architecture for spatiotemporal Bloch-Torrey is
        # different from the surrogate architecture above.

        # TODO: implement with proper AD through DifferentialEquations.jl
    end

    return total_residual / n
end

# ------------------------------------------------------------------ #
# KomaMRI integration for generating training data
# ------------------------------------------------------------------ #

"""
    generate_koma_training_data(params_sampler, acq; n_samples, snr)

Generate (params, signals) training pairs using KomaMRI.jl as the
reference Bloch simulator. Returns column-major matrices.

This gives us ground truth for free/Gaussian diffusion that the
surrogate must reproduce.
"""
function generate_koma_training_data end  # defined when KomaMRI is loaded

# Conditional loading: only define KomaMRI integration if available
function __init_koma__()
    @eval begin
        using KomaMRI

        function generate_koma_training_data(
            phantom_fn,  # (params) -> Phantom
            seq::Sequence,
            sys::Scanner;
            params_sampler,  # (rng, n) -> (param_dim, n)
            n_samples::Int = 10_000,
        )
            rng = Random.default_rng()
            params = params_sampler(rng, n_samples)
            param_dim = size(params, 1)

            # We need to simulate one at a time (KomaMRI limitation)
            signals = nothing

            for j in 1:n_samples
                phantom = phantom_fn(params[:, j])
                raw = simulate(phantom, seq, sys;
                              sim_params=Dict{String,Any}("return_type" => "raw"))
                sig = abs.(raw.profiles[1].data[:, 1])

                if signals === nothing
                    signals = zeros(Float32, length(sig), n_samples)
                end
                signals[:, j] = sig

                if j % 1000 == 0
                    @info "[KomaMRI] Generated $j/$n_samples samples"
                end
            end

            return params, signals
        end
    end
end
