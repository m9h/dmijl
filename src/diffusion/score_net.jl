"""
MLP score network with residual connections and FiLM conditioning.

Port of the JAX MLPScoreNet that achieved 15.5° in autoresearch v2.
Key advantage in Julia: no XLA compilation wall.
"""

function build_score_net(;
    param_dim::Int=10,
    signal_dim::Int=90,
    hidden_dim::Int=512,
    depth::Int=6,
    cond_dim::Int=128,
)
    # Signal encoder: signal → conditioning vector
    signal_encoder = Chain(
        Dense(signal_dim => cond_dim, gelu),
        Dense(cond_dim => cond_dim, gelu),
    )

    # Time encoder: t → conditioning vector
    time_encoder = Chain(
        Dense(1 => cond_dim ÷ 2, Lux.sigmoid_fast),
        Dense(cond_dim ÷ 2 => cond_dim, Lux.sigmoid_fast),
    )

    # Input projection
    input_proj = Dense(param_dim => hidden_dim)

    # Residual blocks with FiLM: each has (linear, gamma_proj, beta_proj)
    blocks = []
    for _ in 1:(depth - 1)
        push!(blocks, (
            linear = Dense(hidden_dim => hidden_dim),
            gamma  = Dense(cond_dim * 2 => hidden_dim),
            beta   = Dense(cond_dim * 2 => hidden_dim),
        ))
    end

    # Output projection
    output_proj = Dense(hidden_dim => param_dim)

    return (;
        signal_encoder,
        time_encoder,
        input_proj,
        blocks = NamedTuple.(blocks),
        output_proj,
    )
end

"""
    score_forward(model, ps, st, theta_t, t, signal)

Forward pass of the score network. Returns (noise_prediction, state).
"""
function score_forward(model, ps, st, theta_t, t, signal)
    # Encode conditioning
    sig_emb, st_sig = model.signal_encoder(signal, ps.signal_encoder, st.signal_encoder)
    t_emb, st_time = model.time_encoder(reshape([t], 1), ps.time_encoder, st.time_encoder)
    cond = vcat(sig_emb, t_emb)

    # Input projection
    h, st_inp = model.input_proj(theta_t, ps.input_proj, st.input_proj)
    h = gelu.(h)

    # Residual blocks with FiLM
    new_block_sts = []
    for (i, block) in enumerate(model.blocks)
        h_in = h
        block_ps = ps.blocks[i]
        block_st = st.blocks[i]

        gamma, st_g = block.gamma(cond, block_ps.gamma, block_st.gamma)
        beta_val, st_b = block.beta(cond, block_ps.beta, block_st.beta)
        h_new, st_l = block.linear(h, block_ps.linear, block_st.linear)

        h = @. h_new * (1 + gamma) + beta_val
        h = gelu.(h)
        h = h .+ h_in  # residual

        push!(new_block_sts, (; linear=st_l, gamma=st_g, beta=st_b))
    end

    # Output
    out, st_out = model.output_proj(h, ps.output_proj, st.output_proj)

    new_st = (;
        signal_encoder = st_sig,
        time_encoder = st_time,
        input_proj = st_inp,
        blocks = Tuple(new_block_sts),
        output_proj = st_out,
    )
    return out, new_st
end
