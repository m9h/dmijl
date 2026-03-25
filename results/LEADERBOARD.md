# Leaderboard: dMRI Microstructure Inference

All results from the development of dmipy-jax and dmijl.

## Flow NPE (Python/JAX) — Ball+2Stick Orientation Error

Target: <3° median Fiber 1 orientation error (Manzano-Patron et al.)

| Date | Config | Fiber 1 | d_stick r | f1 r | Steps | Notes |
|------|--------|---------|-----------|------|-------|-------|
| Mar 19 | Baseline flow (affine, depth=6) | 10.7° | 0.95 | 0.85 | 30k | Starting point |
| Mar 19 | Spline flow + noise fix + label-switching | 6.4° | 0.978 | 0.902 | 30k | All fixes combined |
| Mar 19 | + lr=1e-4, 50k steps | 7.4° | 0.972 | 0.896 | 50k | lr too slow |
| Mar 20 | + lr=3e-4, 100k steps | **4.4°** | 0.983 | 0.924 | 100k | Big improvement |
| Mar 20 | + 200k steps | **3.2°** | 0.986 | 0.935 | 200k | Near target |

### Key findings:
- Training budget dominates: 30k→200k = 6.4°→3.2°
- Wide snr_range=(10,50) helps (regularization)
- Spline transformer > affine
- Label-switching fix in prior essential

## Score-based Posterior (Python/JAX) — Ball+2Stick

| Date | Config | Fiber 1 | d_stick r | Notes |
|------|--------|---------|-----------|-------|
| Mar 20 | e3nn equivariant | 60° | 0.0 | Failed — e3nn layers too restrictive |
| Mar 20 | MLP + SDE sampler | 29° | 0.0 | MLP learns, SDE sampler broken |
| Mar 21 | MLP + DDPM sampler | **15.5°** | 0.977 | DDPM >> SDE >> ODE |
| Mar 21 | + gentle schedule (β_max=5) | **17.6°** | 0.961 | Gentler helps |
| Mar 21 | + v-prediction, 1024-wide, 100k | **14.9°** | 0.983 | Scale up |
| Mar 21 | + spherical coords (θ,φ), 30k | **12.8°** | 0.925 | Parameterization wins |
| Mar 22 | Equivariant head, 30k | 16.1° | 0.978 | Plain MLP still better |

### Key findings:
- e3nn FunctionalLinear can't learn — information can't flow 0e→1o
- DDPM sampler critical (SDE diverges)
- Spherical parameterization: 15.5°→12.8° (eliminates sphere constraint)
- v-prediction helps slightly
- Equivariant output head doesn't help (bottleneck is sampler, not head)

## Bloch-Torrey Surrogate (Julia) — Forward Model Accuracy

Target: <1% median relative error on Ball+2Stick signals

| Date | Config | Med. Error | <1% | <5% | Time | Status |
|------|--------|-----------|-----|-----|------|--------|
| Mar 23 | h256_d6_mse, 10k | 2.49% | 11% | 90% | 262s | — |
| Mar 23 | h256_d6_rel, 10k | 2.32% | 2% | 95% | 527s | — |
| Mar 23 | h256_d6_logcosh, 10k | 2.67% | 8% | 86% | 521s | — |
| Mar 23 | h512_d6_rel, 10k | 93.8% | 0% | 0% | 659s | Diverged |
| Mar 23 | **h256_d6_rel, 50k** | **0.96%** | **53%** | **99%** | 1088s | **PASSED** |

### Key findings:
- Input normalization to [0,1] critical (21.7%→2.3%)
- h256 stable, h512 diverges with relative MSE
- 50k steps crosses <1% threshold
- Surrogate slower than analytical for Ball+Stick (8ms vs 2ms)
- Value is for MC/FEM forward models, not analytical formulas

## D(r) Field Recovery (Julia) — Real WAND Connectom Data

Non-parametric diffusion field from CHARMED data (sub-00395, 7 shells, b=0-6000)

| Date | Version | Config | MD (m²/s) | FA | Loss | Time | Notes |
|------|---------|--------|-----------|-----|------|------|-------|
| Mar 24 | v1 (direction-blind) | tiny 1k | 7.3e-10 | 0.094 | 0.062 | 138s | No gradient dirs |
| Mar 24 | v1 | small 5k | 6.7e-10 | 0.036 | 0.053 | 430s | FA collapsed |
| Mar 24 | v1 | medium 10k | 6.7e-10 | 0.036 | 0.053 | 980s | FA stuck |
| Mar 24 | **v2 (direction-aware)** | tiny 1k | 2.7e-9 | **0.281** | 0.009 | 153s | FA recovered! |
| Mar 24 | v2 | small 5k | 2.75e-9 | 0.279 | 0.005 | 252s | FA stable |
| Mar 24 | v2 | medium 10k | 2.73e-9 | 0.279 | 0.006 | 848s | MD too high |

Expected WM values: MD ~ 0.7e-9 m²/s, FA ~ 0.4-0.7

### Key findings:
- v1→v2: FA 0.036→0.281 by making prediction direction-dependent
- MD overestimated (2.7e-9 vs expected 0.7e-9)
- 8 variant experiments running to fix MD (D-scale, log-loss, regularization)

## Cross-validation

| Test | Result |
|------|--------|
| Microstructure.jl compartments (Cylinder, Zeppelin, Iso, Sphere) | PASS at 1e-13 |
| KomaMRI signal properties (1000 random configs) | 1000/1000 PASS |
| Julia analytical tests (88 tests) | ALL PASS |
| Julia physics invariants (264 tests) | ALL PASS |
| Julia test suite total | 477 PASS, 9 specs |

## DiffEq Sampler Benchmark (Julia)

| Solver | Steps/s |
|--------|---------|
| DiffEq EM (dt=0.01) | 21,708 |
| DiffEq EM (dt=0.002) | 10,716 |
| DiffEq ODE Tsit5 (adaptive) | 8,788 |

## Infrastructure

- **DGX Spark**: Grace CPU (20 ARM cores) + GB10 GPU, 120GB RAM
- **Slurm**: CPU + GPU partitions, fsl_sub integration
- **FSL**: CUDA eddy on GB10 (266 volumes preprocessed in 2h)
- **Trackio**: experiment tracking (HuggingFace)
- **HF Hub**: model checkpoint push/pull
- **KomaMRI.jl**: Bloch simulation validation
- **Microstructure.jl**: cross-validated at machine precision
