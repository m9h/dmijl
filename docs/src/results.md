# Results

All results validated on real data from the **WAND** (Welsh Advanced
Neuroimaging Database) acquired on a Siemens Connectom scanner with
300 mT/m gradients.

## AxCaliber PINN — axon radius from restricted diffusion

**This is the main result.** The Van Gelderen restricted diffusion model
recovers compartment geometry from multi-delta AxCaliber data.

| Parameter | Recovered | Expected WM | Status |
|:----------|:----------|:------------|:-------|
| Axon radius R | **3.15 μm** | 2-5 μm | Correct |
| Intra-cellular fraction | **0.46** | 0.4-0.7 | Correct |
| D intra-cellular | 4.6e-10 m²/s | 1-2e-9 | Low |
| Fiber direction | [0.98, 0.14, 0.14] | — | Dominant x-axis |

- **Data**: sub-00395, 4 AxCaliber acquisitions, Δ = 18/30/42/55 ms
- **b-values**: up to 15,500 s/mm²
- **Training**: 5,000 steps, 168 seconds on DGX Spark Grace CPU
- **Physics**: Van Gelderen GPD approximation for cylinders

### Why this matters

Stejskal-Tanner (S = exp(-bD)) assumes Gaussian diffusion and **cannot**
recover axon radius — it sees only an apparent diffusivity. The Van Gelderen
model captures how signal changes with diffusion time Δ when spins are
trapped inside cylinders: short Δ = unrestricted, long Δ = restricted.
The Δ-dependence encodes the geometry.

## Neural diffusion tensor field — direction-aware fitting

Recovers spatially-varying D(x,y,z) from CHARMED data without assuming
any geometric compartment model.

| Metric | Recovered | Expected WM |
|:-------|:----------|:------------|
| MD | **7.4e-10 m²/s** (0.74 μm²/ms) | ~0.7e-9 |
| FA | **0.42** | 0.4-0.7 |

- **Data**: sub-00395, CHARMED, 7 shells (b = 0-6000 s/mm²), 266 volumes
- **Key insight**: log-space loss critical for correct MD (MSE is dominated
  by b=0 shell)

### Honest label

This is **not** a PINN — no PDE residual is enforced. It uses the
Stejskal-Tanner equation for signal prediction, which assumes Gaussian
diffusion. The "neural" part is the MLP that maps spatial position to
diffusion tensor. This is neural field fitting with physics-based
signal prediction.

## Cross-validation

| Test | Result |
|:-----|:-------|
| Microstructure.jl compartments (Cylinder, Zeppelin, Iso, Sphere) | PASS at 1e-13 |
| KomaMRI signal properties (1000 random configurations) | 1000/1000 PASS |
| Van Gelderen restricted diffusion (112 physics tests) | ALL PASS |
| Julia analytical tests (88 tests) | ALL PASS |
| Julia physics invariants (264 tests) | ALL PASS |

## Full leaderboard

### Flow NPE (SBI4DWI companion project)

The normalizing flow posterior on synthetic Ball+2Stick achieves
**2.8 deg** median orientation error at 300k steps, meeting the
Nottingham paper target (<3 deg).

| Steps | Fiber 1 | d_stick r | f1 r |
|:------|:--------|:----------|:-----|
| 200k | 3.2 deg | 0.986 | 0.935 |
| **300k** | **2.8 deg** | **0.987** | **0.943** |

See [`results/LEADERBOARD.md`](https://github.com/m9h/dmijl/blob/master/results/LEADERBOARD.md)
for the complete history across both DMI.jl and SBI4DWI.
