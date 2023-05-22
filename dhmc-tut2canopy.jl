using TransformVariables, LogDensityProblems, LogDensityProblemsAD,
      DynamicHMC, TransformedLogDensities

using CanopyOptics
using Unitful
using SimpleUnPack
using Statistics, Random, Distributions, LinearAlgebra

using MCMCDiagnosticTools, DynamicHMC.Diagnostics, BenchmarkTools

import ForwardDiff

struct ProspectProblem{TY <: AbstractVector, Tnwl <: Integer}
    "Observations"
    y::TY
    "Wavelengths"
    nwl::Tnwl
end

const opti_c = createLeafOpticalStruct((400.0:1:2500) * u"nm")
function prospect4(N, Cab, Cw, Cm)
    leaf = LeafProspectProProperties{Float64}(N=N, Ccab=Cab, Cw=Cw, Cm=Cm)
    T,R = prospect(leaf, opti_c)
    R
end

prospect4(1.4, 40, 0.01, 0.01)

function (problem::ProspectProblem)(θ)
    @unpack y, nwl = problem
    @unpack N, Cab, Cw, Cm, resid = θ
    priors = [
        Uniform(1.0, 3.0),
        Uniform(0, 150),
        Uniform(0, 0.1),
        Uniform(0, 0.1),
        Uniform(0, 1)
    ]
    𝓁_prior = sum(map(logpdf, priors, θ))
    isfinite(𝓁_prior) || return -Inf
    mod = prospect4(N, Cab, Cw, Cm)
    𝓁_like = loglikelihood(MvNormal(mod, resid * I), obs)
    𝓁_prior + 𝓁_like
end

θ₀ = (N = 1.4, Cab = 40, Cw = 0.01, Cm = 0.01, resid = 0.1)
obs = prospect4(θ₀[[:N, :Cab, :Cw, :Cm]]...)
# Define the problem
p = ProspectProblem(obs, length(obs))
# Try out a likelihood calculation
p(θ₀)

function problem_transformation(p::ProspectProblem)
    as((N = as(Real, 1, Inf), Cab = asℝ₊, Cw = asℝ₊, Cm = asℝ₊, resid = asℝ₊))
end

t = problem_transformation(p)
P = TransformedLogDensity(t, p)
∇P = ADgradient(:ForwardDiff, P)

LogDensityProblems.dimension(P)
LogDensityProblems.logdensity(P, θ₀)

results = map(_ -> mcmc_with_warmup(Random.default_rng(), ∇P, 1000), 1:5)

posterior = transform.(t, eachcol(pool_posterior_matrices(results)))
posterior_β = mean(first, posterior)
posterior_σ = mean(last, posterior)
