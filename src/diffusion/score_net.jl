"""
FiLM-conditioned score network as proper Lux custom layers.

Architecture:
  signal  --> SignalEncoder --> |
                                |--> concat --> CondNet --> (gamma_l, beta_l) per layer
  t       --> SinusoidalEmbed+Dense --> |
  theta_t --> InputProj --> FiLMLayer_1 --> ... --> FiLMLayer_L --> OutputProj --> score

FiLM (Feature-wise Linear Modulation): h' = gamma * activation(W*h + b) + beta
Ref: Perez et al. 2018, "FiLM: Visual Reasoning with a General Conditioning Layer"

Port of the JAX MLPScoreNet that achieved 15.5 deg in autoresearch v2.
Key advantage in Julia: no XLA compilation wall.
"""

# ─── Sinusoidal time embedding (fixed frequencies, not learned) ──────────────

"""
    SinusoidalEmbedding(dim)

Fixed sinusoidal positional embedding for diffusion timestep t in [0, 1].
Maps scalar t -> vector of dimension `dim` using log-spaced frequencies.
Not a trainable layer -- has no parameters.
"""
struct SinusoidalEmbedding <: Lux.AbstractLuxLayer
    dim::Int
end

Lux.initialparameters(::AbstractRNG, ::SinusoidalEmbedding) = NamedTuple()
Lux.initialstates(::AbstractRNG, l::SinusoidalEmbedding) = (;
    frequencies = Float32.(exp.(range(0, log(1000.0f0), length=l.dim ÷ 2)))
)

function (l::SinusoidalEmbedding)(t::AbstractArray, ps, st)
    # t: (1, batch) or (1,)
    freqs = reshape(st.frequencies, :, 1)  # (dim/2, 1)
    args = freqs .* t                       # (dim/2, batch)
    emb = vcat(sin.(args), cos.(args))      # (dim, batch)
    return emb, st
end

# ─── FiLM Layer ──────────────────────────────────────────────────────────────

"""
    FiLMLayer(dense)

Feature-wise Linear Modulation layer. Computes:
    h' = (1 + gamma) .* gelu(dense(x)) .+ beta

where (gamma, beta) are provided as conditioning input.
The dense transform and activation are the "main" path; gamma/beta modulate
the output feature-wise before the residual connection (applied externally).
"""
struct FiLMLayer{D} <: Lux.AbstractLuxContainerLayer{(:dense,)}
    dense::D
end

function (l::FiLMLayer)((x, gamma, beta)::Tuple, ps, st)
    h, st_dense = l.dense(x, ps.dense, st.dense)
    h = gelu.(h)
    # FiLM modulation: scale and shift
    h_mod = (1 .+ gamma) .* h .+ beta
    return h_mod, (; dense = st_dense)
end

# ─── Conditioning Network ────────────────────────────────────────────────────

"""
    ConditioningNet(signal_encoder, time_embed, time_proj, cond_merge, film_projections)

Maps (signal, t) -> conditioning pairs (gamma_l, beta_l) for each FiLM layer.

  signal -> signal_encoder -> sig_emb ─|
                                        |--> cat --> cond_merge --> film_proj_l --> (gamma_l, beta_l)
  t -> sinusoidal_embed -> time_proj -> t_emb ─|

The film_projections field is a NamedTuple of Dense layers (one per FiLM block).
Each Dense maps cond_dim*2 -> hidden_dim*2 and the output is split into
(gamma, beta) each of size hidden_dim.
"""
struct ConditioningNet{SE, TE, TP, CM, FP} <: Lux.AbstractLuxContainerLayer{
    (:signal_encoder, :time_embed, :time_proj, :cond_merge, :film_projections)
}
    signal_encoder::SE
    time_embed::TE
    time_proj::TP
    cond_merge::CM
    film_projections::FP  # NamedTuple of Dense layers, one per FiLM layer
    n_layers::Int
end

function (l::ConditioningNet)(signal, t, ps, st)
    # Encode signal
    sig_emb, st_se = l.signal_encoder(signal, ps.signal_encoder, st.signal_encoder)

    # Encode timestep: sinusoidal embedding + dense projection
    t_emb_raw, st_te = l.time_embed(t, ps.time_embed, st.time_embed)
    t_emb, st_tp = l.time_proj(t_emb_raw, ps.time_proj, st.time_proj)

    # Merge signal and time conditioning
    cond_in = vcat(sig_emb, t_emb)
    cond, st_cm = l.cond_merge(cond_in, ps.cond_merge, st.cond_merge)

    # Project to (gamma, beta) pairs for each FiLM layer
    # Use a recursive helper to build results as tuples (Zygote-friendly)
    gammas, betas, st_fps = _project_film_params(
        l.film_projections, ps.film_projections, st.film_projections,
        cond, Val(l.n_layers),
    )

    new_st = (;
        signal_encoder = st_se,
        time_embed = st_te,
        time_proj = st_tp,
        cond_merge = st_cm,
        film_projections = st_fps,
    )
    return gammas, betas, new_st
end

# Unroll FiLM projection computation at compile time via generated function.
# Each layer_k produces (gamma_k, beta_k) by splitting the Dense output.
@generated function _project_film_params(layers, ps, st, cond, ::Val{N}) where {N}
    layer_syms = [Symbol("layer_$i") for i in 1:N]

    # Build expressions for each layer
    gamma_exprs = []
    beta_exprs = []
    st_exprs = []
    code = Expr[]

    for (i, s) in enumerate(layer_syms)
        gb_sym = Symbol("gb_$i")
        st_sym = Symbol("st_$i")
        hd_sym = Symbol("hd_$i")
        g_sym = Symbol("g_$i")
        b_sym = Symbol("b_$i")

        push!(code, :( ($gb_sym, $st_sym) = layers.$s(cond, ps.$s, st.$s) ))
        push!(code, :( $hd_sym = size($gb_sym, 1) ÷ 2 ))
        push!(code, :( $g_sym = $gb_sym[1:$hd_sym, :] ))
        push!(code, :( $b_sym = $gb_sym[$hd_sym+1:end, :] ))

        push!(gamma_exprs, g_sym)
        push!(beta_exprs, b_sym)
        push!(st_exprs, :( $s = $st_sym ))
    end

    gammas_tuple = Expr(:tuple, gamma_exprs...)
    betas_tuple = Expr(:tuple, beta_exprs...)
    st_nt = Expr(:tuple, st_exprs...)  # NamedTuple construction

    return quote
        $(code...)
        return ($gammas_tuple, $betas_tuple, (; $(st_exprs...)))
    end
end

# ─── Score Network ───────────────────────────────────────────────────────────

"""
    ScoreNetwork(cond_net, input_proj, film_layers, output_proj)

Full score network with FiLM conditioning. Forward pass takes a NamedTuple
`(; theta_t, t, signal)` and returns the score (noise or velocity prediction).

Architecture:
  1. cond_net maps (signal, t) -> [(gamma_l, beta_l) for each layer]
  2. input_proj maps theta_t -> h
  3. Each FiLMLayer: h = film_l((h, gamma_l, beta_l)) + h_residual
  4. output_proj maps h -> score prediction (param_dim)

Both :eps (noise) and :v (velocity) prediction modes are supported --
the prediction target is set in the training loop / sampler, not in the
network itself (the architecture is identical for both).
"""
struct ScoreNetwork{CN, IP, FL, OP} <: Lux.AbstractLuxContainerLayer{
    (:cond_net, :input_proj, :film_layers, :output_proj)
}
    cond_net::CN
    input_proj::IP
    film_layers::FL  # NamedTuple of FiLMLayer
    output_proj::OP
    n_layers::Int
end

function (l::ScoreNetwork)(x::NamedTuple, ps, st)
    theta_t = x.theta_t   # (param_dim, batch) or (param_dim,)
    t       = x.t         # (1, batch) or scalar
    signal  = x.signal    # (signal_dim, batch) or (signal_dim,)

    # Reshape scalar t to (1, batch) if needed
    t_input = _ensure_2d_time(t)

    # Get FiLM conditioning parameters
    gammas, betas, st_cn = l.cond_net(signal, t_input, ps.cond_net, st.cond_net)

    # Input projection
    h, st_ip = l.input_proj(theta_t, ps.input_proj, st.input_proj)
    h = gelu.(h)

    # FiLM residual blocks -- unrolled at compile time for Zygote
    h, st_fl = _apply_film_layers(
        l.film_layers, ps.film_layers, st.film_layers,
        h, gammas, betas, Val(l.n_layers),
    )

    # Output projection
    out, st_op = l.output_proj(h, ps.output_proj, st.output_proj)

    new_st = (;
        cond_net = st_cn,
        input_proj = st_ip,
        film_layers = st_fl,
        output_proj = st_op,
    )
    return out, new_st
end

# Unroll FiLM layer application at compile time.
@generated function _apply_film_layers(layers, ps, st, h, gammas, betas, ::Val{N}) where {N}
    layer_syms = [Symbol("layer_$i") for i in 1:N]

    code = Expr[]
    st_exprs = []

    for (i, s) in enumerate(layer_syms)
        h_res = Symbol("h_res_$i")
        h_film = Symbol("h_film_$i")
        st_sym = Symbol("st_$i")

        push!(code, :( $h_res = h ))
        push!(code, :( ($h_film, $st_sym) = layers.$s(
            (h, gammas[$i], betas[$i]), ps.$s, st.$s
        ) ))
        push!(code, :( h = $h_film .+ $h_res ))

        push!(st_exprs, :( $s = $st_sym ))
    end

    return quote
        $(code...)
        return (h, (; $(st_exprs...)))
    end
end

# Handle both scalar and batched time inputs
_ensure_2d_time(t::Real) = reshape(Float32[t], 1, 1)
_ensure_2d_time(t::AbstractVector) = reshape(t, 1, :)
_ensure_2d_time(t::AbstractMatrix) = t

# ─── Builder ─────────────────────────────────────────────────────────────────

"""
    build_score_net(; param_dim, signal_dim, hidden_dim, depth, cond_dim)

Build a FiLM-conditioned score network as a proper Lux model.

Returns a `ScoreNetwork` that can be initialized with `Lux.setup(rng, model)`
and called as `model(x, ps, st)` where `x = (; theta_t, t, signal)`.

# Arguments
- `param_dim`: dimension of microstructure parameters (e.g. 10 for Ball+2Stick)
- `signal_dim`: dimension of dMRI signal (number of measurements)
- `hidden_dim`: width of hidden layers (default 512)
- `depth`: total number of layers including input/output projections (default 6);
  the number of FiLM residual blocks is `depth - 1`
- `cond_dim`: dimension of conditioning embedding (default 128)
"""
function build_score_net(;
    param_dim::Int = 10,
    signal_dim::Int = 90,
    hidden_dim::Int = 512,
    depth::Int = 6,
    cond_dim::Int = 128,
)
    n_film = depth - 1

    # --- Signal encoder ---
    signal_encoder = Chain(
        Dense(signal_dim => cond_dim, gelu),
        Dense(cond_dim => cond_dim, gelu),
    )

    # --- Time encoder: sinusoidal embedding + dense ---
    time_embed = SinusoidalEmbedding(cond_dim)
    time_proj = Chain(
        Dense(cond_dim => cond_dim, gelu),
        Dense(cond_dim => cond_dim, gelu),
    )

    # --- Conditioning merge: concat sig+time -> shared cond vector ---
    cond_merge = Chain(
        Dense(cond_dim * 2 => cond_dim * 2, gelu),
        Dense(cond_dim * 2 => cond_dim * 2, gelu),
    )

    # --- FiLM projection heads: one per residual block, outputs (gamma, beta) ---
    film_proj_pairs = Pair{Symbol, Any}[]
    for i in 1:n_film
        # Each projects cond -> (gamma, beta) concatenated = 2 * hidden_dim
        push!(film_proj_pairs, Symbol("layer_$i") => Dense(cond_dim * 2 => hidden_dim * 2))
    end
    film_projections = NamedTuple(film_proj_pairs)

    cond_net = ConditioningNet(
        signal_encoder, time_embed, time_proj, cond_merge, film_projections, n_film,
    )

    # --- Input projection ---
    input_proj = Dense(param_dim => hidden_dim)

    # --- FiLM residual blocks ---
    film_layer_pairs = Pair{Symbol, Any}[]
    for i in 1:n_film
        push!(film_layer_pairs, Symbol("layer_$i") => FiLMLayer(Dense(hidden_dim => hidden_dim)))
    end
    film_layers = NamedTuple(film_layer_pairs)

    # --- Output projection ---
    output_proj = Dense(hidden_dim => param_dim)

    return ScoreNetwork(cond_net, input_proj, film_layers, output_proj, n_film)
end

# ─── Convenience wrapper for backward compatibility ──────────────────────────

"""
    score_forward(model::ScoreNetwork, ps, st, theta_t, t, signal)

Convenience wrapper that calls the `ScoreNetwork` with individual arguments
instead of a NamedTuple. Matches the old API signature used by the DiffEq
samplers and other code.
"""
function score_forward(model::ScoreNetwork, ps, st, theta_t, t, signal)
    x = (; theta_t = theta_t, t = t, signal = signal)
    return model(x, ps, st)
end
