# Bayesian PROSPECT inversion with Turing.jl (NUTS, ForwardDiff AD).
# Chains run sequentially.
#
# Usage: julia --project=julia julia/fit_turing.jl --sample lopex_01 [--errmodel iid] [--e1 exact]

include(joinpath(@__DIR__, "src", "ProspectModels.jl"))
using .ProspectModels
using Turing, Distributions, LinearAlgebra, Random
import MCMCChains
using CSV, DataFrames, JSON3, Dates

function parse_args(args)
    out = Dict("sample" => "synthetic", "errmodel" => "iid", "e1" => "exact",
               "version" => "prospectd")
    for i in 1:2:length(args)
        out[replace(args[i], "--" => "")] = args[i + 1]
    end
    return out
end

@model function prospect_iid(inp, e1)
    pN = inp.prior_N
    N ~ truncated(Normal(pN[:mean], pN[:sd]); lower = pN[:lower])
    C ~ product_distribution(LogNormal.(inp.prior_meanlog, inp.prior_sdlog))
    sigma ~ truncated(Normal(0, inp.prior_sigma_scale); lower = 0)
    R = prospect(N, C, inp; e1 = e1)
    inp.obs ~ MvNormal(R, sigma^2 * I)
end

function make_inits(inp, chains, seed)
    ini = read_config("sampling")[:init]
    map(1:chains) do ch
        rng = Xoshiro(seed + ch)
        jit = (2 .* rand(rng, length(inp.prior_meanlog)) .- 1) .* ini[:jitter]
        InitFromParams((N = Float64(ini[:N]), C = exp.(inp.prior_meanlog .+ jit),
                        sigma = Float64(ini[:sigma])))
    end
end

function main(argv)
    args = parse_args(argv)
    inp = load_inputs(args["sample"]; version = args["version"])
    cfg = Dict{Symbol,Any}(pairs(read_config("sampling")[:turing]))
    if get(args, "quick", "false") == "true"  # smoke test
        cfg[:warmup] = 20; cfg[:draws] = 20
    end
    seed = read_config("sampling")[:seed]
    e1 = Symbol(args["e1"])
    model = prospect_iid(inp, e1)
    nchains = cfg[:chains]
    inits = make_inits(inp, nchains, seed)
    sampler = NUTS(cfg[:warmup], cfg[:target_accept])

    # Setup/compilation: a short throwaway run triggers JIT compilation of the
    # model, gradient, and sampler code.
    t_setup = @elapsed sample(Xoshiro(1), model, NUTS(5, 0.8), 5;
                              initial_params = inits[1], progress = false)

    t_sampling = @elapsed chn = sample(
        Xoshiro(seed), model, sampler, MCMCSerial(), cfg[:draws], nchains;
        initial_params = inits, progress = false, chain_type = MCMCChains.Chains,
    )

    nc = length(inp.constituents)
    ndraw = size(chn, 1)
    df = DataFrame(chain = repeat(1:nchains, inner = ndraw), draw = repeat(1:ndraw, nchains))
    flat(sym) = vec(Array(chn[:, sym, :]))
    df.N = flat(Symbol("N"))
    for (j, c) in enumerate(inp.constituents)
        df[!, c] = flat(Symbol("C[$j]"))
    end
    df.sigma = flat(:sigma)

    tag = args["e1"] == "exact" ? args["errmodel"] : "$(args["errmodel"])-$(args["e1"])"
    outdir = joinpath(ProspectModels.ROOT, "results", "fits", "turing", tag, args["sample"])
    mkpath(outdir)
    CSV.write(joinpath(outdir, "draws.csv.gz"), df; compress = true)

    divergences = Int(sum(chn[:, :numerical_error, :]))
    meta = Dict(
        "backend" => "turing",
        "sample_id" => args["sample"],
        "errmodel" => args["errmodel"],
        "e1" => args["e1"],
        "version" => args["version"],
        "n_wavelengths" => length(inp.obs),
        "constituents" => inp.constituents,
        "settings" => cfg,
        "time_setup_s" => t_setup,
        "time_sampling_s" => t_sampling,
        "divergences" => divergences,
        "max_treedepth_hits" => nothing,
        "timestamp" => string(now()),
        "platform" => "Julia $(VERSION)",
        "software" => Dict("Turing" => string(pkgversion(Turing)),
                           "ad_backend" => "ForwardDiff"),
    )
    open(joinpath(outdir, "meta.json"), "w") do io
        JSON3.pretty(io, meta)
    end
    println("Wrote $outdir")
end

main(ARGS)
