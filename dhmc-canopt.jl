using CanopyOptics
using DynamicHMC
using Distributions
using ForwardDiff
using LinearAlgebra
using SimpleUnPack
using Unitful

const opti_c = createLeafOpticalStruct((400.0:1:2500) * u"nm")
function prospect4(N, Cab, Cw, Cm)
    leaf = LeafProspectProProperties{Float64}(N=N, Ccab=Cab, Cw=Cw, Cm=Cm)
    T,R = prospect(leaf, opti_c)
    R
end

struct Model{T}
    observations::T
end

function (m::Model)(θ)
    @unpack N, Cab, Cw, Cm, resid = θ
    priors = [
        Uniform(1.0, 3.0),
        Uniform(0, 150),
        Uniform(0, 0.1),
        Uniform(0, 0.1),
        Uniform(0, 1)
    ]
    log_prior = sum(map(logpdf, priors, θ))
    isfinite(log_prior) || return -Inf
    mod = prospect4(N, Cab, Cw, Cm)
    dist = MvNormal(mod, resid * I)
    log_likelihood = sum(logpdf(dist, m.observations))
    log_prior + log_likelihood
end

const obs = prospect4(1.4, 40, 0.01, 0.01)
m = Model(obs)
ttest = (N=1.4, Cab=40, Cw=0.01, Cm=0.01, resid=0.1)
# Check...
m(ttest)

# Now, transform it
using LogDensityProblems, TransformVariables, TransformedLogDensities
using LogDensityProblemsAD

l = TransformedLogDensity(as((N = asℝ₊, Cab = asℝ₊, Cw = asℝ₊, Cm = asℝ₊, resid = asℝ₊)), m)
LogDensityProblems.logdensity(m, ttest)
∇l = ADgradient(:ForwardDiff, l)
LogDensityProblems.capabilities(∇l)
LogDensityProblems.logdensity_and_gradient(∇l, (1.4, 40, 0.01, 0.01, 0.3))
