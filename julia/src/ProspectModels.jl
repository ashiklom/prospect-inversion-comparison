"""
PROSPECT family in Julia: generic over `Real` (ForwardDiff-compatible) and
non-mutating.

E1 modes: `:exact` uses `SpecialFunctions.expint`, whose derivative
(dE1/dx = -exp(-x)/x) is supplied by DiffRules, so ForwardDiff never
differentiates through the special-function implementation. `:approx` uses
the Swamee & Ohija (2003) closed form.
"""
module ProspectModels

using SpecialFunctions: expint
using CSV, DataFrames, JSON3

export e1_exact, e1_approx, gpm, prospect, load_inputs, read_config

e1_exact(x) = expint(x)

function e1_approx(x)
    a = log((0.56146 / x + 0.65) * (1 + x))
    b = x^4 * exp(7.7 * x) * (2 + x)^3.7
    return (a^-7.7 + b)^-0.13
end

"Transmissivity of a compact layer; assumes k > 0."
function transmissivity(k, e1)
    e = e1 === :exact ? e1_exact(k) : e1_approx(k)
    return max((1 - k) * exp(-k) + k^2 * e, zero(k))
end

"Generalized plate model at one wavelength; returns (R, T)."
function gpm(k, N, talf, t12, t21, e1)
    tau = transmissivity(k, e1)
    ralf = 1 - talf
    r12 = 1 - t12
    r21 = 1 - t21
    denom = 1 - r21^2 * tau^2
    Ta = talf * tau * t21 / denom
    Ra = ralf + r21 * tau * Ta
    t = t12 * tau * t21 / denom
    r = r12 + r21 * tau * t
    D = sqrt(max((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t), zero(r)))
    va = (1 + r^2 - t^2 + D) / (2 * r)
    vb = (1 - r^2 + t^2 + D) / (2 * t)
    vbNN = vb^(N - 1)
    vbNN2 = vbNN^2
    denomx = va^2 * vbNN2 - 1
    Rsub = va * (vbNN2 - 1) / denomx
    Tsub = vbNN * (va^2 - 1) / denomx
    denomy = 1 - Rsub * r
    return (Ra + Ta * Rsub * t / denomy, Ta * Tsub / denomy)
end

"""
    prospect(N, C, inp; e1 = :exact, want_t = false)

PROSPECT reflectance (or `(R, T)` if `want_t`) for structure parameter `N`
and constituent vector `C` (ordered as the columns of `inp.K`).
"""
function prospect(N, C, inp; e1 = :exact, want_t = false)
    k = (inp.K * C) ./ N
    rt = gpm.(k, N, inp.talf, inp.t12, inp.t21, e1)
    R = first.(rt)
    return want_t ? (R, last.(rt)) : R
end

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

read_config(name) = JSON3.read(read(joinpath(ROOT, "config", "$name.json"), String))

"Load fit inputs for one sample (see data/inputs/<version>/<sample>.csv)."
function load_inputs(sample_id; version = "prospectd")
    models = read_config("models")
    constituents = String.(models[version][:constituents])
    d = CSV.read(joinpath(ROOT, "data", "inputs", version, "$sample_id.csv"), DataFrame)
    priors = read_config("priors")
    cp = [priors[:constituents][Symbol(c)] for c in constituents]
    return (
        sample_id = sample_id,
        constituents = constituents,
        wavelength = d.wavelength,
        obs = Float64.(d.R),
        K = Matrix{Float64}(d[:, constituents]),
        talf = Float64.(d.talf), t12 = Float64.(d.t12), t21 = Float64.(d.t21),
        prior_meanlog = Float64[c[:meanlog] for c in cp],
        prior_sdlog = Float64[c[:sdlog] for c in cp],
        prior_N = priors[:N],
        prior_sigma_scale = Float64(priors[:sigma][:scale]),
    )
end

end
