# DMI.jl — Roadmap

## Done

- [x] Composable compartment framework (G1Ball, C1Stick, G2Zeppelin, S1Dot)
- [x] RestrictedCylinder (Soderman), SphereGPD (Murday-Cotts)
- [x] MultiCompartmentModel + ConstrainedModel + fit_mcm
- [x] Watson distribution + DistributedModel
- [x] NODDI-Watson factory with LinkedParameter constraints
- [x] Cross-validation against dmipy-jax (machine precision)
- [x] MCMRSimulator integration + GitHub mirrors
- [x] AxCaliber PINN on real WAND Connectom data (R=3.15 um)
- [x] Neural diffusion tensor field (FA=0.42, MD=7.4e-10)
- [x] Flow NPE at 2.8 deg (Nottingham target met)
- [x] Score-based DDPM posteriors (12.8 deg)
- [x] softplus fix (Julia 1.12 compat)
- [x] Examples migrated to `using DMI`
- [x] Documenter.jl docs + GitHub Actions CI

---

## Phase 1: Core SBI (amortized inference + validation)

### Mixture Density Networks (MDN)
Gaussian mixture posterior — lightweight amortized baseline. ~70 lines of Lux.jl.
Ref: `dmipy_jax/inference/mdn.py`

### Normalizing Flows (NPE)
Neural spline flows for amortized posterior estimation. Achieved 2.8 deg in
JAX — the single best inference result. Port via Bijectors.jl or native
implementation with rational quadratic spline coupling layers.
Ref: `dmipy_jax/inference/flows.py`

### Simulation-Based Calibration (SBC)
Rank histogram diagnostics (Talts et al., 2018). Validates posterior coverage.
Pure Julia, ~200 lines.
Ref: `dmipy_jax/pipeline/sbc.py`

### Conformal Prediction
Distribution-free uncertainty intervals with guaranteed finite-sample coverage.
Works even if MDN/flow is miscalibrated. Pure Julia, algorithmic.
Ref: `dmipy_jax/pipeline/conformal.py`

### Out-of-Distribution Detection
Reconstruction error + Mahalanobis distance + predictive entropy.
Flags unreliable predictions on real data. Pure Julia, ~150 lines.
Ref: `dmipy_jax/pipeline/ood.py`

### Posterior Predictive Checks (PPC)
Compare observed to posterior-predictive distributions.
Detects model misspecification. Pure Julia, ~100 lines.
Ref: `dmipy_jax/pipeline/ppc.py`

---

## Phase 2: Full Bayesian + robustness

### MCMC Inference (NUTS)
Rician log-likelihood + NUTS sampler via AdvancedHMC.jl.
Gold standard for uncertainty quantification + validation reference.
Ref: `dmipy_jax/inference/mcmc.py`

### Variational Inference
Mean-field Gaussian with reparameterization trick. Fast, differentiable.
Ref: `dmipy_jax/inference/variational.py`

### Data Augmentation
Variable SNR sweep, Rician noise matching, label-switching fix for
multi-fiber models. Critical for robust neural posteriors (6.4 deg -> 2.8 deg).
Ref: training scripts in `dmipy_jax/pipeline/simulator.py`

### Deep Ensembles
Train N independent models, average predictions + disagreement for uncertainty.
Wrapper around existing training loop. ~20 lines.
Ref: `dmipy_jax/pipeline/ensemble.py`

---

## Phase 3: Experimental design

### Optimal Experimental Design (OED)
Jacobian-based Fisher information matrix, D-optimality criterion.
Design optimal dMRI protocols to minimize parameter uncertainty.
Ref: `dmipy_jax/design/oed.py`

### Acquisition Optimization
Continuous optimization of b-values, gradient directions, delta, Delta.
Design customized protocols for specific tissue models.
Ref: `dmipy_jax/optimization/acquisition.py`

---

## Phase 4: Clinical models + deployment

### Extended Phase Graph (EPG)
Handle T2-relaxation in multi-echo CPMG sequences. ~400 lines.
Ref: `dmipy_jax/models/epg.py`

### Quantitative Magnetization Transfer (QMT)
Two-pool magnetization exchange with RF saturation. Myelin biomarker.
Ref: `dmipy_jax/models/qmt.py`

### Algebraic Initializers
DTI-based initialization for NLLS fitting. Faster convergence, fewer local minima.
Ref: `dmipy_jax/fitting/algebraic_initializers.py`

### BIDS Data Loaders
Streamlined loading for BIDS, HCP, WAND datasets.
Ref: `dmipy_jax/io/`

### PlaneCallaghan Compartment
Restricted diffusion between parallel planes. Completes the geometry set.
Ref: `dmipy_jax/signal_models/plane_models.py`
