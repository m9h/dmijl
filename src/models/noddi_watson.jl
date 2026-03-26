"""
    noddi_watson(; d_par=1.7e-9, d_iso=3.0e-9, n_grid=300)

Construct a NODDI-Watson model using the composable compartment framework.

Returns a `ConstrainedModel` with 6 free parameters:
  - mu (3): mean fiber orientation (Cartesian unit vector)
  - kappa (1): Watson concentration parameter
  - partial_volume_1 (1): intra-cellular volume fraction (f_intra)
  - partial_volume_2 (1): isotropic volume fraction (f_iso)

Constraints applied:
  - lambda_par fixed at `d_par` for both stick and zeppelin
  - lambda_iso fixed at `d_iso` for ball
  - mu and kappa shared between stick and zeppelin DistributedModels
  - lambda_perp = lambda_par * (1 - f_intra)  (tortuosity)
  - f_extra = 1 - f_intra - f_iso  (volume fraction unity)

Architecture:
  S = f_intra * Watson(Stick) + f_iso * Ball + f_extra * Watson(Zeppelin)

Compartment ordering: (dm_stick, ball, dm_zep) so that:
  - partial_volume_1 = f_intra (free)
  - partial_volume_2 = f_iso (free)
  - partial_volume_3 = f_extra (derived via fraction unity)
"""
function noddi_watson(; d_par::Float64=1.7e-9, d_iso::Float64=3.0e-9, n_grid::Int=300)
    # Build components
    watson = WatsonDistribution(n_grid=n_grid)
    mu0 = [0.0, 0.0, 1.0]
    kappa0 = 10.0

    stick = C1Stick(mu=mu0, lambda_par=d_par)
    dm_stick = DistributedModel(stick, watson, :mu, mu0, kappa0)

    ball = G1Ball(lambda_iso=d_iso)

    zep = G2Zeppelin(mu=mu0, lambda_par=d_par, lambda_perp=d_par * 0.5)
    dm_zep = DistributedModel(zep, watson, :mu, mu0, kappa0)

    # Order: (stick, ball, zep) so partial_volume_1=f_intra, partial_volume_2=f_iso
    mcm = MultiCompartmentModel(dm_stick, ball, dm_zep)

    # --- Dynamically compute flat index mapping ---
    names = parameter_names(mcm)
    card = parameter_cardinality(mcm)

    name_to_idx = Dict{Symbol, Int}()
    idx = 1
    for n in names
        name_to_idx[n] = idx
        idx += card[n]
    end

    # Find the collision-renamed names for the zeppelin's parameters.
    # The MCM appends _<compartment_index> for collisions (zep is compartment 3).
    # We search by prefix to be robust to the exact suffix.
    function find_name(prefix::String)
        for n in names
            s = String(n)
            if s == prefix || (startswith(s, prefix * "_") && all(isdigit, s[length(prefix)+2:end]))
                return n
            end
        end
        error("No parameter matching prefix '$prefix' found in $names")
    end

    # Stick params (first occurrence, no suffix)
    idx_lambda_par_stick = name_to_idx[:lambda_par]
    idx_mu_stick = name_to_idx[:mu]
    idx_kappa_stick = name_to_idx[:kappa]

    # Ball params
    idx_lambda_iso = name_to_idx[:lambda_iso]

    # Zeppelin params (collision-renamed)
    # Find the second lambda_par, mu, kappa (the ones with _N suffix)
    zep_lambda_par_name = nothing
    zep_mu_name = nothing
    zep_kappa_name = nothing
    for n in names
        s = String(n)
        if s != "lambda_par" && startswith(s, "lambda_par_") && !startswith(s, "lambda_par_3_") || s == "lambda_par_3"
            # Match lambda_par_N where N is a digit (collision suffix)
            suffix = s[length("lambda_par_")+1:end]
            if all(isdigit, suffix)
                zep_lambda_par_name = n
            end
        end
        if s != "mu" && startswith(s, "mu_")
            suffix = s[length("mu_")+1:end]
            if all(isdigit, suffix)
                zep_mu_name = n
            end
        end
        if s != "kappa" && startswith(s, "kappa_")
            suffix = s[length("kappa_")+1:end]
            if all(isdigit, suffix)
                zep_kappa_name = n
            end
        end
    end

    zep_lambda_par_name !== nothing || error("Could not find zeppelin's lambda_par in MCM names: $names")
    zep_mu_name !== nothing || error("Could not find zeppelin's mu in MCM names: $names")
    zep_kappa_name !== nothing || error("Could not find zeppelin's kappa in MCM names: $names")

    idx_lambda_par_zep = name_to_idx[zep_lambda_par_name]
    idx_mu_zep = name_to_idx[zep_mu_name]
    idx_kappa_zep = name_to_idx[zep_kappa_name]
    idx_lambda_perp = name_to_idx[:lambda_perp]

    # Constraints (order matters: fixed/linked first, then tortuosity, then fraction unity)
    constraints = (
        # 1. Fix lambda_par on stick
        FixedParameter(:lambda_par, d_par, idx_lambda_par_stick),
        # 2. Fix lambda_par on zeppelin
        FixedParameter(zep_lambda_par_name, d_par, idx_lambda_par_zep),
        # 3. Fix lambda_iso on ball
        FixedParameter(:lambda_iso, d_iso, idx_lambda_iso),
        # 4. Link zep mu -> stick mu (copy 3 elements)
        LinkedParameter(zep_mu_name,
            collect(idx_mu_zep : idx_mu_zep + 2),
            collect(idx_mu_stick : idx_mu_stick + 2)),
        # 5. Link zep kappa -> stick kappa
        LinkedParameter(zep_kappa_name,
            [idx_kappa_zep],
            [idx_kappa_stick]),
        # 6. Tortuosity: lambda_perp = lambda_par_zep * (1 - f_intra)
        TortuosityConstraint(:lambda_perp,
            idx_lambda_perp,
            idx_lambda_par_zep,
            name_to_idx[:partial_volume_1]),
        # 7. Fraction unity: partial_volume_3 = 1 - pv1 - pv2
        VolumeFractionUnity(:partial_volume_3,
            name_to_idx[:partial_volume_3],
            [name_to_idx[:partial_volume_1], name_to_idx[:partial_volume_2]]),
    )

    return ConstrainedModel(mcm, constraints)
end
