# DMI.jl — Open Tasks

## MRIBuilder.jl dependency investigation

MCMRSimulator.jl (Cottaar, FMRIB) depends on MRIBuilder.jl (UUID 691e6122),
which is NOT in the Julia General registry. The GitHub repo
(github.com/MichielCottaar/MRIBuilder.jl) contains only the docs site,
not the source code. The source is likely on FMRIB GitLab
(git.fmrib.ox.ac.uk/ndcn0236/mribuilder.jl).

**Questions to resolve:**
- Is MRIBuilder registered in a custom Julia registry at FMRIB?
- Can MCMRSimulator be used without MRIBuilder (sequence-independent mode)?
- Should we add FMRIB's registry as a secondary Pkg registry?
- Or fork MCMRSimulator to relax the MRIBuilder dependency?

**Workaround:** Our AxCaliber PINN uses Van Gelderen (analytical) instead of
MCMRSimulator (Monte Carlo). KomaMRI.jl handles Bloch simulation for
validation. MCMRSimulator would add value for restricted diffusion ground
truth in arbitrary geometries (meshes, beaded axons).
