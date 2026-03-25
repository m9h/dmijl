using Documenter

# We don't load the full DMI module (heavy deps) — just build from markdown
makedocs(
    sitename = "DMI.jl",
    authors = "Morgan Hough",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://m9h.github.io/dmijl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Results" => "results.md",
        "Methods" => [
            "Compartment Models" => "methods/compartment_models.md",
            "AxCaliber PINN" => "methods/axcaliber_pinn.md",
            "Diffusion Tensor Field" => "methods/tensor_field.md",
            "Score-Based Posterior" => "methods/score_posterior.md",
            "Forward Models" => "methods/forward_models.md",
        ],
        "Validation" => "validation.md",
        "API Reference" => "api.md",
        "Companion: SBI4DWI" => "sbi4dwi.md",
    ],
)

deploydocs(
    repo = "github.com/m9h/dmijl.git",
    devbranch = "master",
)
