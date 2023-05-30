using TransformVariables, LogDensityProblems, LogDensityProblemsAD,
      DynamicHMC, TransformedLogDensities

using SimpleUnPack
# using Parameters
using Statistics, Random, Distributions, LinearAlgebra

using MCMCDiagnosticTools, DynamicHMC.Diagnostics, BenchmarkTools

import ForwardDiff

struct LinearRegressionProblem{TY <: AbstractVector, TX <: AbstractMatrix,
        Tv <: Real}
    "Observations"
    y::TY
    "Covariates"
    X::TX
    "Degrees of freedom for prior"
    v::Tv
end

function (problem::LinearRegressionProblem)(θ)
    @unpack y, X, v = problem
    @unpack β, σ = θ
    ϵ_distribution = Normal(0, σ)
    𝓁_error = mapreduce((y,x) -> logpdf(ϵ_distribution, y - dot(x, β)), +, y, eachrow(X))
    𝓁_σ = logpdf(TDist(v), σ)
    𝓁_β = loglikelihood(Normal(0, 10), β)
    𝓁_error + 𝓁_σ + 𝓁_β
end

N = 100
X = hcat(ones(N), randn(N, 2))
β = [1.0, 2.0, -1.0]
σ = 0.5
y = X*β .+ randn(N) .* σ
p = LinearRegressionProblem(y, X, 1.0)
p((β = β, σ = σ))

function problem_transformation(p::LinearRegressionProblem)
    as((β = as(Array, size(p.X, 2)), σ = asℝ₊))
end

t = problem_transformation(p)
P = TransformedLogDensity(t, p)
∇P = ADgradient(:ForwardDiff, P)

results = map(_ -> mcmc_with_warmup(Random.default_rng(), ∇P, 1000), 1:5)

posterior = transform.(t, eachcol(pool_posterior_matrices(results)))
posterior_β = mean(first, posterior)
posterior_σ = mean(last, posterior)
