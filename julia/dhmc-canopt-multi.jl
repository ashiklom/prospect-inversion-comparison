import Pkg; Pkg.activate(".")
using Revise

using CanopyOptics
using DynamicHMC
using Distributions
using ForwardDiff
using LinearAlgebra
using SimpleUnPack
using Unitful
using Random
using PDMats
using SparseArrays
using LogDensityProblems, TransformVariables, TransformedLogDensities
using LogDensityProblemsAD

using Profile
using BenchmarkTools

using Arrow
using DataFrames, DataFramesMeta

# Load observations
data_path = "/home/ashiklom/projects/prospect-traits-manuscript/data/ecosis-processed/lopex"
metadata_path = "$data_path/metadata.arrow"
spectra_path = "$data_path/spectra.arrow"
metadata = DataFrame(Arrow.Table(metadata_path))
spectra = DataFrame(Arrow.Table(spectra_path))
observation_id = metadata[1, :observation_id]
spectra_sub = @subset spectra :observation_id .== observation_id :spectral_measurement .== "reflectance"
spectra_wide = unstack(spectra_sub, :spectra_id, :value)
const waves = spectra_wide[:, :wavelength_nm]
const obs = Array{Float64}(spectra_wide[:, "lopex_01.001"])

const opti_c = createLeafOpticalStruct(waves * u"nm", method = :interp)
function prospect4(N::T, Cab::T, Cw::T, Cm::T) where {T}
    leaf = LeafProspectProProperties{T}(N=N, Ccab=Cab, Cw=Cw, Cm=Cm)
    _,R = prospect(leaf, opti_c)
    return R
end

"Cholesky lower triangular for AR1 correlation matrix"
function AR1chol(ρ, n, nmax = n)
    R = zeros(eltype(ρ), n, n)
    R[1:nmax, 1] = ρ .^ (0:(nmax-1))
    c = sqrt(1 - ρ^2)
    R2 = c * view(R, :, 1)
    for i in 2:n
        R[i:n, i] = view(R2, 1:(n-i+1))
    end
    return LowerTriangular(R)
end

# https://mathoverflow.net/questions/275831/determinant-of-correlation-matrix-of-autoregressive-model
# det_AR1(ρ, n) = (1 - ρ^2)^(n - 1)
logdet_AR1(ρ, n) = (n - 1) * log(1 - ρ^2)
# https://math.stackexchange.com/questions/975069/the-inverse-of-ar-structure-correlation-matrix-kac-murdock-szeg%C5%91-matrix
inv_AR1(ρ, n) = inv(one(ρ) - ρ^2) * SymTridiagonal([one(ρ), fill(1 + ρ^2, n-2)..., one(ρ)], fill(-ρ, n-1))

ρ = 0.7; n = 5
σ = collect(range(1.0, n, n))
σd = Diagonal(σ)
Ω_L = AR1chol(ρ, n)
Ω = Ω_L * Ω_L'
Σ = σd * Ω * σd

function logpdf_ar1(x, μ, σ, ρ)
    n = size(μ, 1)
    σ⁻¹ = inv(Diagonal(σ))
    s = (x - μ)
    Σ_inv = σ⁻¹ * inv_AR1(ρ, n) * σ⁻¹
    sΣs = s' * Σ_inv * s
    # Ω = Correlation; Σ = Covariance
    # det(Σ) = det(Diagonal(σ²)) * det(Ω)
    # Determinant of diagonal matrix is product of elements.
    # We do this on the variance, not the standard deviation,
    # hence σ², which is 2log(σ) in log space.
    # Product in log space --> sum.
    ldet = logdet_AR1(ρ, n) + 2 * sum(log.(σ))
    logp = n * log(2π) + ldet + sΣs
    return -0.5 * logp
end

# Try out the logdensity
μ = prospect4(1.4, 40.0, 0.01, 0.01)
σ = μ .* 0.1 .+ 0.03
ρ = 0.7
logpdf_ar1(obs, μ, σ, 0.99)

function mylike(obs, μ, α, β, ρ)
    σ = pred .* α .+ β
    logpdf_ar1(obs, μ, σ, ρ)
end

# Explore the consequences of different priors
xin = range(0.01, 0.1, 1000)
lines(xin, x -> mylike(obs, μ, 0, x, ρ)[1])

xin = range(0.0001, 0.02, 1000)
lines(xin, x -> mylike(obs, μ, x, 0.02, ρ)[1])

xin = range(0.99, 1.0-1e-4, 1000)
lines(xin, x -> mylike(obs, μ, 0.005, 0.02, x)[1])

# Target value for α = 0.005
lines(0..0.05, Exponential(0.005))
# Target value for β = 0.02
lines(0..0.1, Exponential(0.02))
# Target value for ρ = 0.999
lines(0.5..0.9999, Beta(12, 1.1))

struct Model{T1, T2}
    waves::T1
    observations::T2
end

function (m::Model)(θ)
    @unpack N, Cab, Cw, Cm, α, β, ρ = θ
    priors = [
        Normal(1.0, 3.0), # N
        Normal(40, 20), # Cab
        Normal(0, 0.1), # Cw
        Normal(0, 0.1), # Cm
        Exponential(0.005), # α
        Exponential(0.02), # β
        Beta(12, 1.1) # ρ
    ]
    log_prior = sum(map(logpdf, priors, θ))
    isfinite(log_prior) || return -Inf
    pred = prospect4(N, Cab, Cw, Cm)
    all(isfinite.(pred)) || return -Inf
    σ = α .* mod .+ β
    log_likelihood = logpdf_ar1(m.observations, pred, σ, ρ)
    log_prior + log_likelihood
end

m = Model(waves, obs)
ttest = (N=1.4, Cab=40.0, Cw=0.01, Cm=0.01,
    α = 0.1, β = 0.1, ρ = 0.7)
# Check...
m(ttest)

# @time mp(1)
# Profile.clear()
# @profile mp(100)
# Profile.print(mincount = 1000)
# # Profile.clear()

# Likelihood itself takes ~245μs
@benchmark m($ttest)

# Profiling results show that the slowest step is:
#     bNm1 = b .^ (N-1)  (in prospect.jl)
# This is just math, so we're about as efficient as we can be.

# Now, transform it
trans = as((
    N = as(Real, 1.0, 6.0),
    Cab = as(Real, 0.0, 300.0),
    Cw = as(Real, 0.0, 0.5),
    Cm = as(Real, 0.0, 0.5),
    α = as(Real, 0.0, 2.0),
    β = as(Real, 0.0, 2.0),
    ρ = as(Real, 0.0, 1.0)
))
l = TransformedLogDensity(trans, m)
∇l = ADgradient(:ForwardDiff, l)

x = randn(LogDensityProblems.dimension(∇l))
@benchmark LogDensityProblems.logdensity_and_gradient($∇l, $x)
# Likelihood with autodiff takes ~2ms --- ~8x slower
# Autodiff of the exponential integral is relatively expensive...but not crazy.

const ∇lc = ∇l
const lcx = randn(LogDensityProblems.dimension(∇lc))
function mt(n)
    x = randn(LogDensityProblems.dimension(∇lc))
    for _ in 1:n
        LogDensityProblems.logdensity_and_gradient(∇lc, x)
    end
end

# Where is the slowdown?
mt(3)
Profile.clear()
@profile mt(5000)
Profile.print(mincount = 300)

# ForwardDiff for expint function is relatively expensive

# Sample
function mysample(n)
    ∇l = ADgradient(:ForwardDiff, l)
    mcmc_with_warmup(Random.default_rng(), ∇l, n;
        reporter = NoProgressReport())
end

# NOTE: Non-linear scaling. The longer it runs, the faster it gets.

# For one observation...
@time r3 = mysample(3)        # 127 seconds
@time r100 = mysample(100)    # 138 seconds
@time r500 = mysample(500)    # 238 seconds
@time r1000 = mysample(1000)  # 337 seconds
@time r5000 = mysample(5000)  # 1328 seconds

using DynamicHMC.Diagnostics
fieldnames(typeof(r1000))
rt = TransformVariables.transform.(trans, eachcol(r1000.posterior_matrix))

lines(getfield.(rt, :β))



hist(first.(rt))
summarize_tree_statistics(r.tree_statistics)
