# Score-Based Posterior

## Method

Denoising score matching trains a neural network to predict the noise
added to parameters at random diffusion times. At inference, the DDPM
(denoising diffusion probabilistic model) discrete reverse process
generates posterior samples.

This gives the **full posterior distribution** over microstructure
parameters — not just point estimates, but uncertainty quantification.

## Architecture

The score network is an MLP with:
- **FiLM conditioning**: signal and time embeddings modulate hidden
  layers via feature-wise linear modulation
- **Residual connections**: gradient flow through deep networks
- **v-prediction**: predicts velocity v = α·ε - σ·θ instead of noise ε,
  giving better gradients at low noise levels

## Best results (Python/JAX)

| Config | Fiber 1 median | Notes |
|:-------|:---------------|:------|
| MLP + DDPM | 15.5° | Baseline |
| + spherical coords (θ,φ) | **12.8°** | Best score-based |
| Flow NPE (for comparison) | **3.2°** | Best overall |

The gap between score (12.8°) and flow (3.2°) is likely due to
DDPM discretization vs exact flow inversion. The flow can be
inverted exactly; DDPM requires many discrete steps.

## Julia implementation

DMI.jl provides the score posterior with native DifferentialEquations.jl
samplers:

```julia
using DMI

# Build score network
model = build_score_net(; param_dim=10, signal_dim=90,
                          hidden_dim=512, depth=6)

# Train via denoising score matching
ps, st, losses = train_score!(model, ps, st;
    simulator_fn = sim_fn,
    prior_fn = prior_fn,
    schedule = VPSchedule(),
    num_steps = 30_000,
    prediction = :v,
)

# Sample posterior (DDPM or DiffEq SDE)
samples = sample_posterior(model, ps, st, signal;
    n_samples = 500, n_steps = 500)

# Or use DifferentialEquations.jl adaptive solver
samples = sample_posterior_diffeq(score_fn, signal, schedule;
    solver = Tsit5())
```

## Honest label

This is **not** a score-based SDE solver in the continuous-time sense.
The training uses denoising score matching (predicting added noise),
and sampling uses the discrete DDPM reverse process with fixed steps.
The DifferentialEquations.jl SDE sampler is available but the DDPM
discrete sampler produces better results in practice.
