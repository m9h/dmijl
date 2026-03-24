"""
GPU/CUDA acceleration utilities for Microstructure.jl.

Provides device selection and data transfer helpers built on Lux's
MLDataDevices integration. CUDA and LuxCUDA are already in Project.toml;
this module makes them usable throughout the training and inference code.

Usage:
    dev = select_device()          # auto-detect GPU or fall back to CPU
    ps = ps |> dev                 # move parameters to device
    batch = to_device(batch, dev)  # explicit transfer helper
"""

using Lux: gpu_device, cpu_device

"""
    select_device(; force_cpu=false)

Auto-detect GPU availability and return the appropriate device transfer
function. Returns `gpu_device()` when CUDA is available and `force_cpu` is
false, otherwise returns `cpu_device()`.

The returned object can be used with the pipe operator:
    dev = select_device()
    ps = ps |> dev
    st = st |> dev
"""
function select_device(; force_cpu::Bool = false)
    if force_cpu
        @info "[Microstructure] Using CPU (force_cpu=true)"
        return cpu_device()
    end

    # gpu_device(; force=false) returns a CPU device when no GPU is found
    # instead of throwing an error.
    dev = gpu_device(; force = false)
    if dev == cpu_device()
        @info "[Microstructure] No GPU detected, using CPU"
    else
        @info "[Microstructure] Using GPU: $dev"
    end
    return dev
end

"""
    to_device(x, dev)

Move arrays, model parameters, or states to the given device.
Equivalent to `x |> dev` but provided as a function for cases where
the pipe operator is inconvenient (e.g., inside closures or generated code).
"""
to_device(x, dev) = dev(x)
