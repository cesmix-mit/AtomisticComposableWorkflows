using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using StaticArrays
using AtomsBase, Unitful, UnitfulAtomic
using InteratomicPotentials, InteratomicBasisPotentials
using CairoMakie
using GalacticOptim, Optim
using JLD
using DPP


# Define 2D points
r = 0.99:0.01:2.5
box = [[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
bcs = [DirichletZero(), DirichletZero(), DirichletZero()]
system = [FlexibleSystem([Atom(:Ar, [0.0, 0.0, 0.0] * u"Å"), Atom(:Ar, [ri, 0.0, 0.0] * u"Å")], 
                        box * u"Å", bcs) for ri in r]

# Get true energies and forces 
ϵ = 0.01034
σ = 1.0 
lj = InteratomicPotentials.LennardJones(ϵ, σ, 4.0, [:Ar])
energies = [InteratomicPotentials.potential_energy(sys, lj) for sys in system]
forces =  [InteratomicPotentials.force(sys, lj)[1][1] for sys in system]

# Define ACE potential
n_body = 2
max_deg = 8
r0 = 1.0
rcutoff = 4.0
wL = 1.0
csp = 1.0
rpi_params = RPIParams([:Ar], n_body, max_deg, wL, csp, r0, rcutoff)

B = reduce(vcat, [evaluate_basis(sys, rpi_params)' for sys in system])
dB = reduce(vcat, [evaluate_basis_d(sys, rpi_params)[1][1, :]' for sys in system])

# Load coefficients
β = load("some_samples_that_should_exist.jld")["β"]


size_inches = (12, 8)
size_pt = 72 .* size_inches
f = Figure(resolution = size_pt, fontsize =8)
ax = Axis(f[1, i], xlabel = "r", ylabel = "Energies (eV)", title = lab)
preds = [B * bi for bi in b[1:150]]

## Plot energies
lines!(ax, r, energies, color=:blue)
for pred in preds
    lines!(ax, r, pred, color = :red, linewidth = 0.1, alpha = 0.1)
end

## or plot errors
# for pred in preds
#     lines!(ax, r, energies - pred, color = :red, linewidth = 0.1, alpha = 0.1)
# end


## Plot forces
ax = Axis(f[2, i], xlabel = "r", ylabel = "Forces (ev/Å))", title = lab)
preds = [dB * bi for bi in b[1:150]]
lines!(ax, r, forces, color=:blue)
for pred in preds
    lines!(ax, r, pred, color = :red, linewidth = 0.1, alpha = 0.1)
end

## or plot errors
# for pred in preds
#     lines!(ax, r, forces - pred, color = :red, linewidth = 0.1, alpha = 0.1)
# end

save("images/2d_comparison_of_IAP_errors_temp.pdf", f)

