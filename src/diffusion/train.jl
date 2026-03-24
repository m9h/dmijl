"""
Training loop for denoising score matching.

No XLA compilation -- Julia JIT compiles incrementally.
Each training step is fast from step 1.

Now uses the unified ScoreNetwork with batched forward passes
instead of per-sample loops.

Supports GPU acceleration via an optional `device` keyword argument.
Pass `device = select_device()` to auto-detect and use CUDA when available.
"""

using Optimisers, Zygote

function train_score!(
    model::ScoreNetwork, ps, st;
    simulator_fn,        # (rng, theta_norm) -> signals
    prior_fn,            # (rng, n) -> theta_norm
    schedule::VPSchedule = VPSchedule(),
    num_steps::Int = 50_000,
    batch_size::Int = 512,
    learning_rate::Float64 = 3e-4,
    print_every::Int = 1000,
    prediction::Symbol = :eps,  # :eps or :v
    device = cpu_device(),
)
    # Move model parameters and state to device
    ps = ps |> device
    st = st |> device

    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), ps)
    rng = Random.default_rng()
    losses = Float64[]

    t0 = time()

    for step in 1:num_steps
        # Sample batch (generated on CPU, then transferred)
        theta = prior_fn(rng, batch_size)       # (param_dim, batch)
        signal = simulator_fn(rng, theta)        # (signal_dim, batch)

        # Sample diffusion time and noise
        t_batch = rand(rng, Float32, 1, batch_size) .* 0.9999f0 .+ 1f-5
        eps = randn(rng, Float32, size(theta))

        # Move batch data to device
        theta  = theta  |> device
        signal = signal |> device
        t_batch = t_batch |> device
        eps     = eps     |> device

        # Forward diffusion
        sig_rate = sqrt.(alpha_bar.(Ref(schedule), t_batch))
        noise_rate = sqrt.(1.0f0 .- alpha_bar.(Ref(schedule), t_batch))
        theta_t = sig_rate .* theta .+ noise_rate .* eps

        # Target
        if prediction == :v
            target = sig_rate .* eps .- noise_rate .* theta
        else
            target = eps
        end

        # Gradient step -- batched forward pass through ScoreNetwork
        (loss, st), grads = Zygote.withgradient(ps) do p
            x = (; theta_t = theta_t, t = t_batch, signal = signal)
            pred_batch, new_st = model(x, p, st)
            loss = mean((pred_batch .- target).^2)
            return loss, new_st
        end

        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
        push!(losses, loss)

        if step % print_every == 0 || step == 1
            elapsed = time() - t0
            rate = step / elapsed
            println("[Score] step $step/$num_steps  loss=$(round(loss, digits=4))  " *
                    "($(round(rate, digits=0)) steps/s)")
        end
    end

    elapsed = time() - t0
    println("[Score] Done. $num_steps steps in $(round(elapsed, digits=1))s")

    # Move results back to CPU for downstream use
    ps_cpu = ps |> cpu_device()
    st_cpu = st |> cpu_device()
    return ps_cpu, st_cpu, losses
end
