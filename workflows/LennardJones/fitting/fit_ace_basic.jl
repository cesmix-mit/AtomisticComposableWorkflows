using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using StaticArrays
using AtomsBase, Unitful, UnitfulAtomic
using InteratomicPotentials, InteratomicBasisPotentials
using CairoMakie
using GalacticOptim, Optim
using JLD
using DPP

include("load_data.jl")
include("fitting/fitting_tools.jl")

# Load data
systems, energies, forces, stresses = load_data("LJ_MD/1/data.xyz"; max_entries = 20000);


## Split into training, testing
training_set, testing_set = train_test_split( [systems, energies, forces, stresses] );
training_system, training_energy, training_forces, training_stress = training_set;
test_system, test_energy, test_forces, test_stress = testing_set;

## Create RPI Basis (2body, 8 polynomial degree)
# This creates the ACE potential
n_body = 2  # 2-body
max_deg = 8 # 8 degree polynomials
r0 = 1.0 # minimum distance between atoms
rcutoff = 4.0 # cutoff radius 
wL = 1.0 # Defaults, See ACE.jl documentation 
csp = 1.0 # Defaults, See ACE.jl documentation
rpi_params = RPIParams([:Ar], n_body, max_deg, wL, csp, r0, rcutoff)

## Get energy and force data 
# ee - energies 
# ff - forces 
# B - ACE descriptors 
# dB - Gradient of B (descriptors w.r.t. forces)

## Get all data
# ee, ff, B, dB = get_data(set, rpi_params)
## Get random subset
batch_size = 20
ee, ff, B, dB = get_data_random_batch(training_set, batch_size, rpi_params)
A = [B; dB] # Design matrix 
b = [ee; ff] # righthand side

## Solve for states, given covariance 
# σe = ..., σf = ... 
# Q = Diagonal( [σe .+ 0.0*ee; σf .+ 0.0*ff])
# Qβ = Symmetric( inv(A' * inv(Q) * A) )
# β = Qβ * (A' * inv(Q) * b)


## Solve jointly for states
x0 = [-1.0; -1.0; A\b]
p = [A, ee, ff]
jnll = OptimizationFunction(joint_neg_log_likelihood, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(jnll, x0, p)
jnll_solution = solve(prob, BFGS()).u
β = jnll_solution[3:end]
σe = jnll_solution[1]
σf = jnll_solution[2]
jnll_cov = GalacticOptim.ForwardDiff.hessian(xx -> joint_neg_log_likelihood(xx, p), jnll_solution)



# Compute Testing Errors
batch_size = 60
ee, ff, B, dB = get_data_random_batch(testing_set, batch_size, rpi_params)

Atest = [B; dB]
btest = [ee; ff]

errors = Atest * β - btest ## Do analysis


# Export ACE 
# Remember to change parameters.ace ACE1.julia -> ACE
# using ACE1
# using JuLIP
# function β_to_file(β, rpi_params, directory_name)
#         try
#             mkdir("ACE_MD/$(directory_name)/")
#         catch
        
#     end
#     basis = get_rpi(rpi_params)
#     IP = JuLIP.MLIPs.combine(basis, β)
#     ACE1.Export.export_ace("ACE_MD/$(directory_name)/parameters.ace", IP)

#     # Make adjustment for package name ACE1.jl -> ACE.jl
#     fff = readlines("ACE_MD/$(directory_name)/parameters.ace")
#     fff[10] = "radbasename=ACE.jl.Basic"
#     open("ACE_MD/$(directory_name)/parameters.ace", "w") do io
#         for l in fff
#             write(io, l*"\n")
#         end
#     end
# end

