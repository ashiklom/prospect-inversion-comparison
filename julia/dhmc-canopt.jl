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

using LogDensityProblems, TransformVariables, TransformedLogDensities
using LogDensityProblemsAD

using Profile
using BenchmarkTools

const opti_c = createLeafOpticalStruct((400.0:1:2500) * u"nm")
function prospect4(N::T, Cab::T, Cw::T, Cm::T) where {T}
    leaf = LeafProspectProProperties{T}(N=N, Ccab=Cab, Cw=Cw, Cm=Cm)
    _,R = prospect(leaf, opti_c)
    return R
end

struct Model{T}
    observations::T
end

function (m::Model)(θ)
    @unpack N, Cab, Cw, Cm, resid = θ
    priors = [
        Normal(1.0, 3.0),
        Normal(40, 20),
        Normal(0, 0.1),
        Uniform(0, 0.1),
        Uniform(0, 1)
    ]
    log_prior = sum(map(logpdf, priors, θ))
    isfinite(log_prior) || return -Inf
    mod = prospect4(N, Cab, Cw, Cm)
    dist = MvNormal(mod, resid * I)
    log_likelihood = logpdf(dist, m.observations)
    log_prior + log_likelihood
end

const obs = prospect4(1.4, 40.0, 0.01, 0.01)
m = Model(obs)
ttest = (N=1.4, Cab=40.0, Cw=0.01, Cm=0.01, resid=0.1)
# Check...
m(ttest)

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
