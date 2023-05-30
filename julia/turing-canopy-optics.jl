using CanopyOptics
using Unitful
using Turing
using Distributions
using ForwardDiff
using LinearAlgebra

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

@model function fitprospect(obs)
    # Priors
    N ~ Uniform(1.1, 3)
    Cab ~ Uniform(10, 100)
    Cw ~ Uniform(0.001, 0.02)
    Cm ~ Uniform(0.001, 0.02)
    resid ~ InverseGamma(1, 0.2)
    mod = prospect4(N, Cab, Cw, Cm)
    obs ~ MvNormal(mod, resid * I)
end

function my_fit_prospect(n)
    sample(fitprospect(obs), NUTS(), n)
end
my_fit_prospect(10) # Precompile
my_fit_prospect(500) # 13-25 seconds
my_fit_prospect(5000) # 157.65 seconds

using DynamicHMC
function dhmc_prospect(n)
    sample(fitprospect(obs), DynamicNUTS(), n)
end

dhmc_prospect(10) # Precompile
dhmc_prospect(500) # 62.57 seconds (but a lot of that was before first step)
dhmc_prospect(5000) # 201.94 seconds

using Profile, PProf
@profile my_fit_prospect(10)
@profile my_fit_prospect(500)
@profile my_fit_prospect(5000)

function tracker_fit_prospect(n)
    sample(fitprospect(obs), NUTS{Turing.TrackerAD}(), n)
end
tracker_fit_prospect(10)  # Precompile
tracker_fit_prospect(500) # 52.49 seconds

using ReverseDiff
function reverse_fit_prospect(n)
    Turing.setadbackend(:reversediff)
    Turing.setrdcache(true)
    sample(fitprospect(obs), NUTS(), n)
end
reverse_fit_prospect(10)  # Precompile
reverse_fit_prospect(500) # 

using Zygote
function zygote_fit_prospect(n)
    Turing.setadbackend(:zygote)
    Turing.setrdcache(false)
    sample(fitprospect(obs), NUTS{}(), n)
end

zygote_fit_prospect(10)
zygote_fit_prospect(500)


pprof()

@profview my_fit_prospect(10)
@profview my_fit_prospect(500)


# Works, but somewhat slow. 5000 iterations took 659.27 seconds (~11 minutes).
# 500 steps takes ~35 seconds.
chain_f = sample(fitprospect(obs), NUTS(), 500)

# Both of these work. MLE produces an estimate much closer to the truth.
using Optim
tmap = optimize(fitprospect(obs), MAP())
tmle = optimize(fitprospect(obs), MLE())

using DynamicHMC
# Startup takes a while...but then, sampling takes ~37 seconds 
chain_dhmc = sample(fitprospect_mv(obs), DynamicNUTS(), 500)
plot(chain_dhmc)

# Trying with a Normal-InverseWishart prior
rΦ = Matrix(Diagonal(fill(0.2, length(obs))))
riw = InverseWishart(length(obs), rΦ)

@model function fitprospect_mv(obs, ::Type{T} = Float64) where {T}
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
#= function cond_resid(c)
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
) =#
