"""
Training loop for denoising score matching.

No XLA compilation — Julia JIT compiles incrementally.
Each training step is fast from step 1.
"""

using Optimisers, Zygote

function train_score!(
    model, ps, st;
    simulator_fn,        # (rng, theta_norm) -> signals
    prior_fn,            # (rng, n) -> theta_norm
    schedule::VPSchedule = VPSchedule(),
    num_steps::Int = 50_000,
    batch_size::Int = 512,
    learning_rate::Float64 = 3e-4,
    print_every::Int = 1000,
    prediction::Symbol = :eps,  # :eps or :v
)
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), ps)
    rng = Random.default_rng()
    losses = Float64[]

    t0 = time()

    for step in 1:num_steps
        # Sample batch
        theta = prior_fn(rng, batch_size)       # (param_dim, batch)
        signal = simulator_fn(rng, theta)        # (signal_dim, batch)

        # Sample diffusion time and noise
        t_batch = rand(rng, Float32, 1, batch_size) .* 0.9999f0 .+ 1f-5
        eps = randn(rng, Float32, size(theta))

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

        # Gradient step
        (loss, st), grads = Zygote.withgradient(ps) do p
            pred_batch = similar(theta)
            current_st = st
            for j in 1:batch_size
                pred_j, current_st = score_forward(
                    model, p, current_st,
                    @view(theta_t[:, j]),
                    t_batch[1, j],
                    @view(signal[:, j]),
                )
                pred_batch[:, j] = pred_j
            end
            loss = mean((pred_batch .- target).^2)
            return loss, current_st
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
    return ps, st, losses
end
