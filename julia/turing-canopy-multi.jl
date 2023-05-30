using CanopyOptics
using Unitful
using Turing
using Distributions
using ForwardDiff
using LinearAlgebra

# LJK Workaround
using TransformVariables
using PDMats

# See Turing documentation: https://turing.ml/dev/docs/using-turing/guide

# Simulate an observation
const opti_c = createLeafOpticalStruct((400.0:1:2500) * u"nm")
const leaf_c = LeafProspectProProperties{Float64}(N=1.4, Ccab = 40, Cw = 0.01, Cm = 0.01)
const _, obs = prospect(leaf_c, opti_c)
# using Plots
# plot(obs)

function prospect4(N::T, Cab::T, Cw::T, Cm::T) where {T}
    leaf = LeafProspectProProperties{T}(N=N, Ccab=Cab, Cw=Cw, Cm=Cm)
    _,R = prospect(leaf, opti_c)
    return R
end


# LKJ prior:
#   - For native support, waiting on some PRs https://github.com/TuringLang/Turing.jl/issues/1870
#   - Workaround: https://discourse.julialang.org/t/singular-exception-with-lkjcholesky/85020

# @model function fitprospect(obs)
#     # Priors
#     N ~ truncated(Normal(1.3, 1.0); lower = 1.0)
#     Cab ~ truncated(Normal(40, 30); lower = 0)
#     Cw ~ truncated(Normal(0.01, 0.01); lower = 0)
#     Cm ~ truncated(Normal(0.01, 0.01); lower = 0)
#     a ~ Normal(0.0, 10.0)
#     b ~ Normal(0.0, 10.0)
#     mod = prospect4(N, Cab, Cw, Cm)
#     σ ~ filldist(truncated(Cauchy(0., 5.); lower = 0), nobs)
#     # ρ ~ LKJ(nobs, 1.0)
#     # LKJ workaround
#     trans = CorrCholeskyFactor(nobs)
#     R_tilde ~ filldist(Flat(), TransformVariables.dimension(trans))
#     R_U, logdetJ = transform_and_logjac(trans, R_tilde)
#     F = Cholesky(R_U)
#     Turing.@addlogprob! logpdf(LKJCholesky(nobs, 1.0), F) + logdetJ
#     Σ_L = LowerTriangular(collect((σ .+ 1e-6) .* R_U'))
#     Σ = PDMat(Cholesky(Σ_L))
#     obs ~ MvNormal(mod, Σ)
# end
# sample(fitprospect(obs), NUTS(0.8), 10)

# Manual implementation of a Normal-InverseWishart
# Wishart prior
const nobs = length(obs)
const V₀ = Matrix(Diagonal(fill(0.2, nobs)))
const m₀ = nobs
const IW₀ = InverseWishart(m₀, V₀)

function IW_post(s, m₀, V₀)
    """Wishart posterior, given known mean. `s = xᵢ - μ`"""
    m₁ = m₀ + nobs
    ss = s * s'
    V₁ = V₀ + ss
    return InverseWishart(m₁, V₁)
end

@model function fitprospect(obs)
    # Turing priors
    N ~ truncated(Normal(1.3, 1.0); lower = 1.0)
    Cab ~ truncated(Normal(40, 30); lower = 0)
    Cw ~ truncated(Normal(0.01, 0.01); lower = 0)
    Cm ~ truncated(Normal(0.01, 0.01); lower = 0)
    # PROSPECT...
    mod = prospect4(N, Cab, Cw, Cm)
    # Manual Gibbs sampling step for residual
    v = 1 + nobs
    s = obs - mod
    ss = s * s'
    Φ = rΦ + ss
    # Manual prior on residual
    Turing.@addlogprob! logpdf(riw, Σ)
    obs ~ MvNormal(mod, Σ)
end


my_fit_prospect(10) # Precompile
my_fit_prospect(500) # 13-17 seconds
