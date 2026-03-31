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
- [x] ROMEO + MriResearchTools phase processing pipeline (WAND 7T MEGRE validated)
- [x] Phase 3a OED: FIM, D/A/E-optimality, CRLB, protocol comparison (34 tests)
- [x] Phase 3b OED: protocol optimization (Optim.jl L-BFGS-B), standard recipes (45 tests)
- [x] SpinDoctor.jl modernized for Julia 1.12 and integrated as FEM oracle
- [x] QMT MTR maps computed for sub-08033 (10 MT conditions, normalized by MT-off)
- [ ] QMT generalized Bloch fitting awaits CUBRIC TRF/pulse shape confirmation

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
- [x] **Phase 3a**: Fisher Information Matrix (Gaussian + Rician), D/A/E-optimality,
  CRLB, hardware constraints, electrostatic directions, `compare_protocols()`.
  34 tests. Ref: Alexander 2008 (DOI:10.1002/mrm.21646)
- [x] **Phase 3b**: Gradient-based protocol optimization via Optim.jl L-BFGS-B.
  Standard protocol recipes (HCP, NODDI, axon diameter). 45 tests.
- [x] **Phase 3c**: Bayesian OED — PCE lower bound and variational upper bound on
  Expected Information Gain. MDN serves as q(theta|y,xi) for the variational
  bound. 57 tests total. Ref: Foster et al. 2019 (arXiv:1903.05480)
- [x] **Phase 3d**: Sequential/adaptive design — greedy FIM-based D-optimality
  maximization over candidate measurements. Ref: Foster et al. 2021 (arXiv:2103.02438)

---

## Phase 4: Clinical models + deployment

### Extended Phase Graph (EPG)
Handle T2-relaxation in multi-echo CPMG sequences. ~400 lines.
BlochSimulators.jl (van der Heide et al. MRM 2024) provides GPU-accelerated
EPG with MR Fingerprinting dictionary generation.
Ref: `dmipy_jax/models/epg.py`

### Quantitative Magnetization Transfer (QMT)
Two-pool magnetization exchange with RF saturation. Myelin biomarker.
MRIgeneralizedBloch.jl (Asslaender et al. MRM 2022) implements the generalized
Bloch model. WAND sub-08033 ses-02 has 10 QMT volumes (3 flip angles x 7 MT
offsets, dense near 1-3 kHz) + full mcDESPOT VFA data ready to process.
Ref: `dmipy_jax/models/qmt.py`

### SpinDoctor.jl validation oracle
FEM Bloch-Torrey PDE solutions for restricted diffusion. Complements
MCMRSimulator (Monte Carlo) and KomaMRI (Bloch sequences). Quantifies Van
Gelderen GPD approximation error at actual WAND timing (delta=11ms).
Future: differentiable FEM for ReMiDi/Spinverse-style mesh-from-signal
inversion (Khole et al. 2025, BioMedAI-UCSC).

### Algebraic Initializers
DTI-based initialization for NLLS fitting. Faster convergence, fewer local minima.
Ref: `dmipy_jax/fitting/algebraic_initializers.py`

### BIDS Data Loaders
Streamlined loading for BIDS, HCP, WAND datasets.
Note: WAND mcDESPOT VFA data (ses-02) has per-volume flip angles missing
from BIDS sidecars — need CUBRIC protocol documentation to resolve.
Ref: `dmipy_jax/io/`

### PlaneCallaghan Compartment
Restricted diffusion between parallel planes. Completes the geometry set.
Ref: `dmipy_jax/signal_models/plane_models.py`
