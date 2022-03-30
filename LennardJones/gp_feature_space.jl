using AtomsBase
using CairoMakie
using InteratomicPotentials 
using InteratomicBasisPotentials
using LinearAlgebra 
using Random
using StaticArrays
using Statistics 
using UnitfulAtomic
using Unitful 

include("load_data.jl")

systems, energies, forces, stresses = load_data("LJ_MD/1/data.xyz"; max_entries = 20000);

# Split into training, testing
N_train = 10
N_batch = 40
N_test  = 100
rand_list = randperm(length(systems))
train_index, test_index = rand_list[1:N_train], rand_list[N_train+1:N_train+1+N_test]
train_systems, train_energies, train_forces, train_stress = systems[train_index], energies[train_index], forces[train_index], stresses[train_index];
test_systems, test_energies, test_forces, test_stress = systems[test_index], energies[test_index], forces[test_index], stresses[test_index];

## Create RPI Basis (2body, 8 polynomial degree)
n_body = 2
max_deg = 8
r0 = 1.0
rcutoff = 4.0
wL = 1.0
csp = 1.0
rpi_params = RPIParams([:Ar], n_body, max_deg, wL, csp, r0, rcutoff)

## Calculate descriptors 
B = [evaluate_basis(i, rpi_params) for i in train_systems]
dB = [evaluate_basis_d(i, rpi_params) for i in train_systems]
A = [hcat(B...)'; vcat(vcat(dB...)...)]

# Calculate Errors
f = vcat(vcat(train_forces...)...)
b = [train_energies; f]
q = [var(train_energies) .+ 0.0*train_energies; var(f) .+ 0.0 * f]
pred_coeff = A' * (Diagonal(q) \ A) \ (A' * (Diagonal(q) \ b) )

B_test = [evaluate_basis(i, rpi_params) for i in test_systems]
error = abs.( hcat(B_test...)' * pred_coeff - test_energies)
μ = mean(B)
C = cov(B)
mahal(x) = sqrt( (x - μ)' * ( C \ (x - μ) ) )

dists = mahal.(B)


struct GP 
    ℓ 
    α
    σ
    X
end

function kernel(x, y, gp::GP)
    K = -0.5 * (x - y)' * (C \ (x-y)) / gp.ℓ^2
    exp(K) * gp.α^2
end

function build(gp::GP)
    K = hcat( [[ kernel(x, xx, gp) for x in gp.X] for xx in gp.X]... )
    K += gp.σ * I(length(gp.X))
    Symmetric(K)
end

function build(Y, gp::GP)
    Kyx = hcat( [[ kernel(x, y, gp) for y in Y] for x in gp.X]... )
    Kyy = hcat( [[ kernel(x, y, gp) for x in Y] for y in Y]... )
    Kyx, Symmetric(Kyy)
end

function predict(x, Y, gp)
    Kxx = build(gp)
    Kyx, Kyy = build(Y, gp)
    y_pred = Kyx * (Kxx \ x)
    σ_pred = diag( Kyy - Kyx * (Kxx \ Kyx') )
    return y_pred, σ_pred 
end

function ll(x, Y, y, gp)
    y_pred, σ_pred = predict(x, Y, gp)
    σ_pred = @. max(0.0, σ_pred) + 1e-8
    0.5 * (y - y_pred)' * (Diagonal(σ_pred) \ (y - y_pred) ) + 0.5 * sum(log.(σ_pred))
end

function fit(p, params)
    ℓ = exp(p[1])
    α = exp(p[2])
    σ = exp(p[3])
    gp = GP(ℓ, α, σ, B_test[1:50])

    ll(error[1:50], B_test, error, gp)
end

f = OptimizationFunction(fit, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(f, [3.0, 0.1, 1e-4], [])
sol = solve(prob,BFGS())



size_in_inches = (10, 8)
size_in_pt = size_in_inches .* 72
f = CairoMakie.Figure(resolution = size_in_pt, fontsize = 12)
axes1 = CairoMakie.Axis(f[1, 1], title="Mahal Dist", xlabel="α", ylabel="ℓ")

CairoMakie.contourf!(axes1, α, ℓ, dll, levels = 40)

CairoMakie.resize_to_layout!(f)
CairoMakie.save("gp_lj_ace.pdf", f)








# ## Load LJ MD 
# lj_systems, _, _, _ = load_data("LJ_MD/TEMP/data.xyz"; max_entries = 100000);
# ace_systems, _, _, _ = load_data("ACE_MD/coeff_1/TEMP/data.xyz"; max_entries = 100000);

# B_lj = [evaluate_basis(i, rpi_params) for i in lj_systems]
# μ_lj = mean(B_lj)
# C_lj = cov(B_lj)
# mahal(x) = sqrt( (x - μ_lj)' * ( C_lj \ (x - μ_lj) ) )
# lj_dists = mahal.(B_lj)

# B_ace = [evaluate_basis(i, rpi_params) for i in ace_systems]
# μ_ace = mean(B_ace)
# C_ace = cov(B_ace)
# ace_dists = mahal.(B_ace)

# size_in_inches = (10, 8)
# size_in_pt = size_in_inches .* 72
# f = CairoMakie.Figure(resolution = size_in_pt, fontsize = 12)
# axes1 = CairoMakie.Axis(f[1, 1], title="Mahal Dist", xlabel="time", ylabel="Mahal Dist")

# t = 50:50:Int(1.5E6)
# t = t * 0.005
# # a = moving_average(lj_dists, 1000)
# # b = moving_average(ace_dists, 1000)
# CairoMakie.lines!(axes1, lj_dists, color = (:red, 0.75))
# CairoMakie.lines!(axes1, ace_dists, color = (:blue, 0.75))

# CairoMakie.resize_to_layout!(f)
# CairoMakie.save("md_lj_ace.pdf", f)


# moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]
