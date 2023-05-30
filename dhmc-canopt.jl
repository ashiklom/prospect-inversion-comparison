using CanopyOptics
using DynamicHMC
using Distributions
using ForwardDiff
using LinearAlgebra
using SimpleUnPack
using Unitful
using Random

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

# Now, transform it
using LogDensityProblems, TransformVariables, TransformedLogDensities
using LogDensityProblemsAD

trans = as((
    N = as(Real, 1.0, 3.0),
    Cab = as(Real, 0.0, 100.0),
    Cw = as(Real, 0.0, 0.1),
    Cm = as(Real, 0.0, 0.1),
    resid = as(Real, 0.0, 1.0)
))
l = TransformedLogDensity(trans, m)
∇l = ADgradient(:ForwardDiff, l)

# Test out 
using BenchmarkTools
x = randn(LogDensityProblems.dimension(∇l))
@benchmark LogDensityProblems.logdensity_and_gradient($∇l, $x)

# Sample
r = mcmc_with_warmup(Random.default_rng(), ∇l, 100)

results = map(_ -> mcmc_with_warmup(Random.default_rng(), ∇l, 1000), 1:3)

LogDensityProblems.capabilities(∇l)
LogDensityProblems.logdensity_and_gradient(∇l, (1.4, 40, 0.01, 0.01, 0.3))
