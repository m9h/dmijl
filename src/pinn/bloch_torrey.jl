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

using Lux, Random, Optimisers, Zygote, Statistics, LinearAlgebra

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
    train_surrogate!(model, ps, st, data_fn; n_steps, batch_size, lr, device)

Train surrogate on (params, signal) pairs from a reference simulator.

`data_fn(rng, n)` should return `(params, signals)` each of shape
`(dim, n)` — column-major batches.

Supports GPU acceleration via the `device` keyword argument.
Pass `device = select_device()` to auto-detect CUDA.
"""
function train_surrogate!(
    model, ps, st, data_fn;
    n_steps::Int = 20_000,
    batch_size::Int = 512,
    learning_rate::Float64 = 1e-3,
    print_every::Int = 1000,
    loss_type::Symbol = :log_cosh,
    device = cpu_device(),
)
    # Move model parameters and state to device
    ps = ps |> device
    st = st |> device

    opt_state = Optimisers.setup(Adam(learning_rate), ps)
    rng = Random.default_rng()
    losses = Float64[]

    t0 = time()

    for step in 1:n_steps
        # Generate training batch from reference simulator (CPU), then transfer
        params_batch, signals_batch = data_fn(rng, batch_size)
        params_batch  = params_batch  |> device
        signals_batch = signals_batch |> device

        # Compute loss and gradients
        (loss, st), grads = Zygote.withgradient(ps) do p
            pred, new_st = model(params_batch, p, st)
            if loss_type == :log_cosh
                # Log-cosh loss: smooth L1, better for relative errors
                diff = pred .- signals_batch
                l = mean(log.(cosh.(diff .* 10.0f0)) ./ 10.0f0)
            elseif loss_type == :relative_mse
                # Relative MSE: upweights low-signal regions
                l = mean(((pred .- signals_batch) ./ max.(signals_batch, 0.01f0)).^2)
            else
                l = mean((pred .- signals_batch).^2)
            end
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

    # Move results back to CPU for downstream use
    ps_cpu = ps |> cpu_device()
    st_cpu = st |> cpu_device()
    return ps_cpu, st_cpu, losses
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
    _eval_M_components(model, ps, st, inp)

Evaluate the PINN surrogate and return (M_re, M_im) as separate scalars.
`inp` is a 4-vector [t; x₁; x₂; x₃].
"""
function _eval_M_components(model, ps, st, inp::AbstractVector)
    out, _ = model(reshape(inp, :, 1), ps, st)
    # out shape: (2, 1) — real and imaginary parts of magnetization
    return out[1, 1], out[2, 1]
end

"""
    _make_scalar_fn(model, ps, st, component::Symbol)

Return a closure `f(inp) -> scalar` that evaluates either the real or
imaginary component of the magnetization. This scalar function is
suitable for `Zygote.gradient` and `Zygote.forwarddiff`.
"""
function _make_scalar_fn(model, ps, st, component::Symbol)
    if component == :re
        return inp -> begin
            out, _ = model(reshape(inp, :, 1), ps, st)
            out[1, 1]
        end
    else  # :im
        return inp -> begin
            out, _ = model(reshape(inp, :, 1), ps, st)
            out[2, 1]
        end
    end
end

"""
    pde_loss(residual, model, ps, st, t, x, D, T2)

Compute the PDE residual loss at collocation points using Zygote AD.

The PINN surrogate takes spatiotemporal input [t; x] (4-vector) and outputs
complex magnetization M as two real outputs: [M_re; M_im].

Derivatives ∂M/∂t and ∇²M are computed via Zygote:
- First derivatives use `Zygote.gradient` on scalar-valued functions
- Second derivatives (for the Laplacian) use `Zygote.forwarddiff` for the
  inner derivative (forward-mode via ForwardDiff) composed with reverse-mode
  Zygote for the outer, avoiding unsupported nested reverse-mode AD.

Residual: R = ∂M/∂t - D·∇²M + iγ(G·x)M + M/T₂
PDE loss: mean(|R|²)
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
    gamma = Float32(res.gamma)
    total_residual = 0.0f0

    for i in 1:n
        ti  = t[i]
        xi  = x[:, i]
        Di  = D[i]
        T2i = T2[i]

        # Build the spatiotemporal input vector: [t; x₁; x₂; x₃]
        inp = vcat([Float32(ti)], Float32.(xi))

        # Scalar functions for each component of M
        f_re = _make_scalar_fn(model, ps, st, :re)
        f_im = _make_scalar_fn(model, ps, st, :im)

        # --- Evaluate M at (ti, xi) ---
        M_re, M_im = _eval_M_components(model, ps, st, inp)

        # --- ∂M/∂t and ∇M via AD (full gradient of the 4-input) ---
        # grad_re[j] = ∂M_re/∂inp[j], where inp = [t, x1, x2, x3]
        grad_re = Zygote.gradient(f_re, inp)[1]
        grad_im = Zygote.gradient(f_im, inp)[1]

        # ∂M/∂t: first element of gradient
        dMre_dt = grad_re[1]
        dMim_dt = grad_im[1]
        dM_dt = ComplexF32(dMre_dt, dMim_dt)

        # --- ∇²M (Laplacian) via second derivatives w.r.t. x ---
        # For each spatial dimension k ∈ {1,2,3}, compute ∂²M/∂xₖ².
        # Use Zygote.forwarddiff for the inner derivative (forward-mode)
        # composed with Zygote.gradient for the outer (reverse-mode).
        # This mixed-mode approach avoids nested reverse-mode AD.
        laplacian_re = 0.0f0
        laplacian_im = 0.0f0

        for k in 1:3
            idx = k + 1  # offset by 1 because inp[1] = t

            # ∂²M_re/∂xₖ²: outer Zygote over inner ForwardDiff
            d2Mre_dxk2 = Zygote.gradient(inp[idx]) do xk_val
                # Inner derivative via forward-mode AD
                Zygote.forwarddiff(xk_val) do xk_inner
                    inp_mod = vcat(inp[1:idx-1], [xk_inner], inp[idx+1:end])
                    f_re(inp_mod)
                end
            end[1]

            d2Mim_dxk2 = Zygote.gradient(inp[idx]) do xk_val
                Zygote.forwarddiff(xk_val) do xk_inner
                    inp_mod = vcat(inp[1:idx-1], [xk_inner], inp[idx+1:end])
                    f_im(inp_mod)
                end
            end[1]

            laplacian_re += something(d2Mre_dxk2, 0.0f0)
            laplacian_im += something(d2Mim_dxk2, 0.0f0)
        end

        laplacian = ComplexF32(laplacian_re, laplacian_im)

        # --- Gradient field at this time point ---
        G = res.gradient_fn(ti)

        # --- Bloch-Torrey residual ---
        # R = ∂M/∂t - D·∇²M + iγ(G·x)M + M/T₂
        M = ComplexF32(M_re, M_im)
        precession = im * gamma * dot(G, xi) * M
        relaxation = M / T2i
        R = dM_dt - Di * laplacian + precession + relaxation

        total_residual += abs2(R)
    end

    return total_residual / n
end

# ------------------------------------------------------------------ #
# Combined PINN training: supervised + PDE residual
# ------------------------------------------------------------------ #

"""
    train_pinn!(model, ps, st, data_fn, residual;
                n_steps, batch_size, n_colloc, lr, lambda_pde,
                colloc_sampler, D_fn, T2_fn, device)

Train a PINN surrogate with a combined loss:

    L = L_data + λ · L_pde

where L_data is supervised MSE on (params, signal) pairs from MCMRSimulator
(or another reference simulator), and L_pde is the Bloch-Torrey PDE residual
evaluated at randomly sampled collocation points.

Supports GPU acceleration via the `device` keyword argument.
Pass `device = select_device()` to auto-detect CUDA.

Note: The PDE residual computation (`pde_loss`) uses per-sample loops with
Zygote AD for second derivatives, which does not currently benefit from GPU
acceleration. The supervised data loss path is fully GPU-accelerated; the
collocation data is kept on CPU for the PDE residual.

# Arguments
- `model`: Lux model (PINN variant taking spatiotemporal input)
- `ps, st`: model parameters and state
- `data_fn(rng, n)`: returns `(params, signals)` each `(dim, n)`
- `residual::BlochTorreyResidual`: PDE residual configuration

# Keyword Arguments
- `n_steps`: number of training steps (default: 20_000)
- `batch_size`: data batch size (default: 256)
- `n_colloc`: number of collocation points per step (default: 128)
- `learning_rate`: Adam learning rate (default: 1e-3)
- `lambda_pde`: weight for PDE loss term (default: 0.1)
- `colloc_sampler(rng, n)`: returns `(t, x)` collocation points;
    `t` is `(n,)` and `x` is `(3, n)` (default: uniform in [0,1]×[-1,1]³)
- `D_fn(rng, n)`: returns diffusivity vector `(n,)` (default: 2.0e-9)
- `T2_fn(rng, n)`: returns T2 vector `(n,)` (default: 80e-3)
- `print_every`: logging interval (default: 1000)
- `device`: device transfer function (default: `cpu_device()`)
"""
function train_pinn!(
    model, ps, st, data_fn,
    residual::BlochTorreyResidual;
    n_steps::Int = 20_000,
    batch_size::Int = 256,
    n_colloc::Int = 128,
    learning_rate::Float64 = 1e-3,
    lambda_pde::Float64 = 0.1,
    colloc_sampler = nothing,
    D_fn = nothing,
    T2_fn = nothing,
    print_every::Int = 1000,
    device = cpu_device(),
)
    # Move model parameters and state to device
    ps = ps |> device
    st = st |> device

    opt_state = Optimisers.setup(Adam(learning_rate), ps)
    rng = Random.default_rng()
    losses_data = Float64[]
    losses_pde  = Float64[]
    losses_total = Float64[]

    # Default collocation sampler: t ∈ [0, 1], x ∈ [-1, 1]³
    if colloc_sampler === nothing
        colloc_sampler = (rng, n) -> begin
            t_c = rand(rng, Float32, n)
            x_c = 2.0f0 .* rand(rng, Float32, 3, n) .- 1.0f0
            return t_c, x_c
        end
    end

    # Default diffusivity: free water at 37°C ≈ 2.0e-9 m²/s
    if D_fn === nothing
        D_fn = (rng, n) -> fill(2.0f-9, n)
    end

    # Default T2: ~80 ms (white matter at 3T)
    if T2_fn === nothing
        T2_fn = (rng, n) -> fill(80.0f-3, n)
    end

    # We need CPU copies of ps/st for the PDE residual (per-sample AD loops)
    _cpu = cpu_device()

    t0 = time()

    for step in 1:n_steps
        # --- Sample supervised data (CPU) then transfer ---
        params_batch, signals_batch = data_fn(rng, batch_size)
        params_batch_dev  = params_batch  |> device
        signals_batch_dev = signals_batch |> device

        # --- Sample collocation points (stay on CPU for PDE residual) ---
        t_c, x_c = colloc_sampler(rng, n_colloc)
        D_c  = D_fn(rng, n_colloc)
        T2_c = T2_fn(rng, n_colloc)

        # --- Combined gradient step ---
        (loss_total, (st, l_data, l_pde)), grads = Zygote.withgradient(ps) do p
            # Supervised loss (on device)
            pred, new_st = model(params_batch_dev, p, st)
            l_data = mean((pred .- signals_batch_dev).^2)

            # PDE residual loss — uses per-sample loops with nested AD,
            # so we transfer params to CPU for this computation.
            p_cpu  = p      |> _cpu
            st_cpu = new_st |> _cpu
            l_pde = pde_loss(residual, model, p_cpu, st_cpu, t_c, x_c, D_c, T2_c)

            l_total = l_data + Float32(lambda_pde) * l_pde
            return l_total, (new_st, l_data, l_pde)
        end

        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
        push!(losses_data, l_data)
        push!(losses_pde, l_pde)
        push!(losses_total, loss_total)

        if step % print_every == 0 || step == 1
            elapsed = time() - t0
            rate = step / elapsed
            @info "[PINN] step $step/$n_steps  " *
                  "L_total=$(round(loss_total, sigdigits=4))  " *
                  "L_data=$(round(l_data, sigdigits=4))  " *
                  "L_pde=$(round(l_pde, sigdigits=4))  " *
                  "($(round(rate, digits=0)) steps/s)"
        end
    end

    elapsed = time() - t0
    @info "[PINN] Done. $n_steps steps in $(round(elapsed, digits=1))s"

    # Move results back to CPU for downstream use
    ps_cpu = ps |> _cpu
    st_cpu = st |> _cpu
    return ps_cpu, st_cpu, (; data=losses_data, pde=losses_pde, total=losses_total)
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
