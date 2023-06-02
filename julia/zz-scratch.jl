μ = collect(1.0:1.0:10.0)
σ = collect(range(1, 5, 10))
km = kernelpdmat(SqExponentialKernel(), 1:10)
Σ = X_A_Xt(km, Diagonal(σ))
dist = MvNormal(μ, Σ)

Y₀ = rand(dist, 50)
mean(Y₀, dims = 2)
cov(Y₀')

Y = μ .+ rand(MvNormal(Σ), 50)
mean(Y, dims = 2)
cov(Y')

Y = μ .+ (rand(MvNormal(km), 50) .* σ)
mean(Y, dims = 2)
cov(Y')

logpdf(dist, Y₀)
logpdf(MvNormal(Σ), Y₀ .- μ)
logpdf(MvNormal(km), (Y₀ .- μ) ./ σ)
rand(MvNormal(μ, Σ), 5)
μ + rand(MvNormal(Σ), 5)
μ + rand(MvNormal(km), 1)

(Y - μ) ./ σ

X = collect(1.0:1.0:10.0)
Xm = reshape(X, :, 1)

ρ = 0.01
k = SqExponentialKernel() ∘ ScaleTransform(ρ) + ConstantKernel(;c=X)
kernelmatrix(k, X)

k = k ∘ LinearTransform(Xm) ∘ ScaleTransform(ρ)
K = kernelmatrix(k, X)

function mvn_sample(K)
    L = cholesky(K + 1e-6 * I)
    v = randn(size(waves, 1), 5)
    f = L.L * v
    return f
end

K = kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(0.1), waves)
fig = Figure()
ax = Axis(fig[1, 1])
for y in eachcol(mvn_sample(K))
    lines!(ax, waves, y)
end

################################################################################

fig = Figure()
ax = Axis(fig[1,1])
for x in [0.1, 0.5, 1.0, 2.0]
    lines!(ax, range(0, 2, 100), Exponential(x), label = "x=$x")
end
axislegend(ax)

################################################################################

#
# Σ = σΩσ
# inv(Σ) = inv(σ) * inv(Ω) * inv(σ)?
σdinv = inv(σd)
s_inv = σdinv * inv_AR1(ρ, n) * σdinv
Σ_inv = inv(Σ)

AR1det(0.7, 5)
det(Ω)
prod(σ.^2) * det(Ω)
det(Σ)

