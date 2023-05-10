using CanopyOptics
using Unitful
using Turing
using Optim
using Distributions
using ForwardDiff
using Plots
using StatsPlots
using FillArrays
using LinearAlgebra

# See Turing documentation: https://turing.ml/dev/docs/using-turing/guide

# Basic test of CanopyOptics itself
opti = createLeafOpticalStruct((400.0:1:2500) * u"nm")
leaf = LeafProspectProProperties{Float64}(Ccab = 30.0)
T,R = prospect(leaf, opti)

plot(R; label = "Reflectance")
plot!(1 .- T; label = "Transmittance")

const opti_c = createLeafOpticalStruct((400.0:1:2500) * u"nm")
function prospect4(N, Cab, Cw, Cm)
    leaf = LeafProspectProProperties{Float64}(N=N, Ccab=Cab, Cw=Cw, Cm=Cm)
    T,R = prospect(leaf, opti_c)
    R
end

# Need this for ForwardAD for some reason...
function prospect4(
    N::ForwardDiff.Dual,
    Cab::ForwardDiff.Dual,
    Cw::ForwardDiff.Dual,
    Cm::ForwardDiff.Dual,
)
    prospect4(N.value, Cab.value, Cw.value, Cm.value)
end

# Simulate an observation
const obs = prospect4(1.4, 40, 0.01, 0.01)
plot(obs)

@model fitprospect(obs) = begin
    # Priors
    N ~ Uniform(1.1, 3)
    Cab ~ Uniform(10, 100)
    Cw ~ Uniform(0.001, 0.02)
    Cm ~ Uniform(0.001, 0.02)
    resid ~ InverseGamma(1, 0.2)
    mod = prospect4(N, Cab, Cw, Cm)
    for i in 1:length(obs)
        obs[i] ~ Normal(mod[i], resid)
    end
end

# Very fast, but extremely inefficient. Algorithm may not be adaptive-enough?
chain = sample(fitprospect(obs), MH(), 5000)
plot(chain)

# Very efficient sampling, but very slow. Need to figure out how to speed it
# up. Might be related to automatic differentiation and/or types.
chain2 = sample(fitprospect(obs), NUTS(), 100)
plot(chain2)

# Both of these work. MLE produces an estimate much closer to the truth.
tmap = optimize(fitprospect(obs), MAP())
tmle = optimize(fitprospect(obs), MLE())

@model fitprospect_mv(obs) = begin
    # Priors
    N ~ Uniform(1.1, 3)
    Cab ~ Uniform(10, 100)
    Cw ~ Uniform(0.001, 0.02)
    Cm ~ Uniform(0.001, 0.02)
    resid ~ InverseGamma(1, 0.2)
    mod = prospect4(N, Cab, Cw, Cm)
    obs ~ MvNormal(mod, resid * I)
end

# This works a tiny bit faster than univariate version, but is still slow
chain_mv = sample(fitprospect_mv(obs), NUTS(), 100)
plot(chain_mv)

# This hangs indefinitely...
using DynamicHMC
chain_dhmc = sample(fitprospect_mv(obs), DynamicNUTS(), 100)
plot(chain_dhmc)

# Trying with a Normal-InverseWishart prior
rΦ = Matrix(Diagonal(fill(0.2, length(obs))))
riw = InverseWishart(length(obs), rΦ)

@model fitprospect_mv2(obs) = begin
    # Priors
    N ~ Uniform(1.1, 3)
    Cab ~ Uniform(10, 100)
    Cw ~ Uniform(0.001, 0.02)
    Cm ~ Uniform(0.001, 0.02)
    resid ~ InverseWishart(length(obs), rΦ)
    # Likelihood
    mod = prospect4(N, Cab, Cw, Cm)
    obs ~ MvNormal(mod, resid)
end

# This fails 
chain_mv2 = sample(fitprospect_mv2(obs), NUTS(), 100)
plot(chain_mv)

# Trying Gibbs sampling --- NUTS for PROSPECT, conjugate prior for residual
function cond_resid(c)
    ν = 1 + length(obs)
    mod = prospect4(c.N, c.Cab, c.Cw, c.Cm)
    s = obs - mod
    ss = s * s'
    Φ = rΦ + ss
    return InverseWishart(ν, Φ)
end

# Maybe works, but is incredibly slow
chain_mv3 = sample(
    fitprospect_mv2(obs),
    Gibbs(
        #= NUTS(:N, :Cab, :Cw, :Cm), =#
        MH(:N), MH(:Cab), MH(:Cw), MH(:Cm),
        GibbsConditional(:resid, cond_resid)
    ),
    100
)
