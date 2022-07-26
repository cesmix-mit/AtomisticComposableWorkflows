using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using StaticArrays
using AtomsBase, Unitful, UnitfulAtomic
using InteratomicPotentials, InteratomicBasisPotentials
using DPP
using CairoMakie
using GalacticOptim, Optim

include("load_data.jl")
include("fitting/fitting_tools.jl")

systems, energies, forces, stresses = load_data("LJ_MD/1/data.xyz"; max_entries = 20000);

## Use curated subset
curated_indices = [1:10:3500; 6000:10:7000; 9000:10:10000]
csystems, cenergies, cforces, cstresses = systems[curated_indices], energies[curated_indices], forces[curated_indices], stresses[curated_indices];

# Split into training, testing
training_set, testing_set = train_test_split( [csystems, cenergies, cforces, cstresses] );
training_system, training_energy, training_forces, training_stress = training_set;
test_system, test_energy, test_forces, test_stress = testing_set;

## Create RPI Basis (2body, 8 polynomial degree)
n_body = 2
max_deg = 8
r0 = 1.0
rcutoff = 4.0
wL = 1.0
csp = 1.0
rpi_params = RPIParams([:Ar], n_body, max_deg, wL, csp, r0, rcutoff)

# Get DPP Mode 
batch_size = 20
dpp = DPPKernel(training_set, rpi_params)
# ee, ff, B, dB = get_data_dpp_mode(dpp, batch_size)
# ee, ff, B, dB = get_data_dpp_batch(dpp, 20)

A = [B; dB]
b = [ee;ff]


probs = vec(inclusion_prob(dpp.L))

## Solve jointly for states
x0 = [-1.0; -1.0; A\b]
p = [A, ee, ff]
jnll = OptimizationFunction(joint_neg_log_likelihood, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(jnll, x0, p)
jnll_solution = solve(prob, BFGS()).u
β = jnll_solution[3:end]
σe = jnll_solution[1]
σf = jnll_solution[2]
jnll_cov = inv(GalacticOptim.ForwardDiff.hessian( x-> joint_neg_log_likelihood(x, p), jnll_solution)[3:end, 3:end])

## Solve for variance only 
x0 = [-1.0, -1.0]
p = [A, ee, ff]
vnll = OptimizationFunction(var_neg_log_likelihood, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(vnll, x0, p)
vnll_solution = solve(prob, BFGS()).u
vnll_cov = GalacticOptim.ForwardDiff.hessian(var_neg_log_likelihood, vnll_solution)

## Solve for the states with given covariance 
# σe = 0.033452202499903233; σf = 8.190871248224453e-7; # Previously fitted 
Q = Diagonal( [σe .+ 0.0*ee; σf .+ 0.0*ff])

Qβ = Symmetric( inv(A' * inv(Q) * A) )
βm = Qβ * (A' * inv(Q) * b)
β = [βm + rand(MvNormal(Qβ)) for i = 1:500]


# ## Export ACE 
# # Remember to change parameters.ace ACE1.julia -> ACE
# using ACE1
# using JuLIP
# for (i, β) in enumerate(β_batches)
#     try
#         mkdir("ACE_MD/coeff_$i/")
#     catch
        
#     end
#     basis = get_rpi(rpi_params)
#     IP = JuLIP.MLIPs.combine(basis, β)
#     ACE1.Export.export_ace("ACE_MD/coeff_$i/parameters.ace", IP)
# end

# include("ACE_MD/run_ace_md.jl")



