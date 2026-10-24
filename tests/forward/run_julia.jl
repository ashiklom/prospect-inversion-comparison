# Evaluate the Julia PROSPECT implementation on the shared test cases.
include(joinpath(@__DIR__, "..", "..", "julia", "src", "ProspectModels.jl"))
using .ProspectModels, CSV, DataFrames, JSON3

root = ProspectModels.ROOT
cases = JSON3.read(read(joinpath(root, "tests", "forward", "cases.json"), String))
coef = CSV.read(joinpath(root, "data", "coefficients", "prospectd.csv"), DataFrame)
cons = String.(cases[:constituents])
inp = (K = Matrix{Float64}(coef[:, cons]), talf = coef.talf, t12 = coef.t12, t21 = coef.t21)
for e1 in (:exact, :approx)
    out = DataFrame()
    for c in cases[:cases]
        R, T = prospect(Float64(c[:N]), Float64.(c[:C]), inp; e1 = e1, want_t = true)
        append!(out, DataFrame(case = String(c[:name]), wavelength = coef.wavelength, R = R, T = T))
    end
    CSV.write(joinpath(root, "tests", "forward", "out", "julia_$(e1).csv"), out)
end
println("julia forward tests written")
