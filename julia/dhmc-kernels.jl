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

using KernelFunctions, Distances

using Profile
using BenchmarkTools

using Arrow
using DataFrames, DataFramesMeta

using GLMakie

# Load observations
data_path = "/home/ashiklom/projects/prospect-traits-manuscript/data/ecosis-processed/lopex"
metadata_path = "$data_path/metadata.arrow"
spectra_path = "$data_path/spectra.arrow"
metadata = DataFrame(Arrow.Table(metadata_path))
spectra = DataFrame(Arrow.Table(spectra_path))
observation_id = metadata[1, :observation_id]
spectra_sub = @subset spectra :observation_id .== observation_id :spectral_measurement .== "reflectance"
spectra_wide = unstack(spectra_sub, :spectra_id, :value)
const waves = Array{Float64}(spectra_wide[:, :wavelength_nm])
const obs = Array{Float64}(spectra_wide[:, "lopex_01.001"])

const opti_c = createLeafOpticalStruct(waves * u"nm", method = :interp)
function prospect4(N::T, Cab::T, Cw::T, Cm::T) where {T}
    leaf = LeafProspectProProperties{T}(N=N, Ccab=Cab, Cw=Cw, Cm=Cm)
    _,R = prospect(leaf, opti_c)
    return R
end

mtest = prospect4(1.4, 40.0, 0.01, 0.01)
ϵ_test = obs - mtest
lines(waves, ϵ_test)

ϵ_norm = ϵ_test ./ mtest
lines(waves, ϵ_norm)

ρ = 1 / 10
k = SqExponentialKernel() ∘ ScaleTransform(ρ)
Ω = kernelpdmat(k, waves)
logpdf(MvNormal(Ω), ϵ_norm)

struct Model{T1, T2}
    waves::T1
    observations::T2
end

function (model::Model)(θ)
    @unpack N, Cab, Cw, Cm, α, β, ρ = θ
    priors = [
        Normal(1.0, 3.0),
        Normal(40, 20),
        Normal(0, 0.1),
        Normal(0, 0.1),
        Exponential(0.1),
        Exponential(0.1),
        Exponential(0.1)
    ]
    log_prior = sum(map(logpdf, priors, θ))
    isfinite(log_prior) || return -Inf
    pred = prospect4(N, Cab, Cw, Cm)
    all(isfinite.(pred)) || return -Inf
    σ = α .* pred .+ β
    Δ = (pred .- model.observations) ./ σ
    K = SqExponentialKernel() ∘ ScaleTransform(ρ)
    Ω = kernelpdmat(K, model.waves)
    log_likelihood = logpdf(MvNormal(Ω), Δ)
    log_prior + log_likelihood
end

MvNormal

m = Model(waves, obs)
ttest = (N=1.4, Cab=40.0, Cw=0.01, Cm=0.01,
    α = 0.1, β = 0.1, ρ = 0.7)
# Check...
m(ttest)

function mp(n)
    m = Model(waves, obs)
    ttest = (N=1.4, Cab=40.0, Cw=0.01, Cm=0.01,
        α = 0.1, β = 0.1, ρ = 0.7)
    for _ in 1:n
        m(ttest)
    end
end

@time mp(1)
Profile.clear()
@profile mp(100)
Profile.print(mincount = 1000)
# Profile.clear()

@benchmark m($ttest)

# Profiling results show that the slowest step is:
#     bNm1 = b .^ (N-1)  (in prospect.jl)
# This is just math, so we're about as efficient as we can be.

# Now, transform it
trans = as((
    N = as(Real, 1.0, 3.0),
    Cab = as(Real, 0.0, 100.0),
    Cw = as(Real, 0.0, 0.1),
    Cm = as(Real, 0.0, 0.1),
    resid = as(Real, 0.0, 1.0)
))
l = TransformedLogDensity(trans, m)
∇l = ADgradient(:ForwardDiff, l)

# Likelihood itself takes ~245μs
@benchmark m($ttest)

x = randn(LogDensityProblems.dimension(∇l))
@benchmark LogDensityProblems.logdensity_and_gradient($∇l, $x)
# Likelihood with autodiff takes ~1.7ms --- ~7x slower
# Autodiff of the exponential integral is relatively expensive...but not crazy.

const ∇lc = ∇l
const lcx = randn(LogDensityProblems.dimension(∇lc))
function mt(n)
    x = randn(LogDensityProblems.dimension(∇lc))
    for i in 1:n
        LogDensityProblems.logdensity_and_gradient(∇lc, x)
    end
end

# Where is the slowdown?
mt(3)
Profile.clear()
@profile mt(5000)
Profile.print(mincount = 500)

# ForwardDiff for expint function is relatively expensive

# Sample
function mysample(n)
    mcmc_with_warmup(Random.default_rng(), ∇lc, n;
        reporter = NoProgressReport())
end

# NOTE: Non-linear scaling. The longer it runs, the faster it gets.

@time r3 = mysample(3)        # 45 seconds
@time r100 = mysample(100)    # 66 seconds (1.5 / sec)
@time r500 = mysample(500)    # 122 seconds (4.1 / sec)
@time r1000 = mysample(1000)  # 170 seconds (5.8 / sec)
@time r = mysample(5000)      # 266 seconds (18 / sec)

using DynamicHMC.Diagnostics
rt = transform.(trans, eachcol(r.posterior_matrix[:,2500:end]))
summarize_tree_statistics(r.tree_statistics)
