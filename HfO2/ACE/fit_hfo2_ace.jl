using AtomsBase
using InteratomicPotentials 
using InteratomicBasisPotentials
using LinearAlgebra 
using Random
using StaticArrays
using Statistics 
using StatsBase
using UnitfulAtomic
using Unitful 
using BenchmarkTools
#using ThreadsX #julia --threads 4

include("load_data.jl")

#systems, energies, forces, stresses = load_data("HfO2_relax_1000.xyz")
systems, energies, forces, stresses = load_data("a-Hfo2-300K-NVT.extxyz")

# Split into training, testing
n_systems = 2000 #length(systems)
n_train = floor(Int, n_systems * 0.8)
n_test  = n_systems - n_train

rand_list = randperm(n_systems)
train_index, test_index = rand_list[1:n_train], rand_list[n_train+1:n_systems]
train_systems, train_energies, train_forces, train_stress =
                             systems[train_index], energies[train_index],
                             forces[train_index], stresses[train_index]
test_systems, test_energies, test_forces, test_stress =
                             systems[test_index], energies[test_index],
                             forces[test_index], stresses[test_index]

# Create RPI Basis 
n_body = 5
max_deg = 6
r0 = 1.0 # ( rnn(:Hf) + rnn(:O) ) / 2.0 ?
rcutoff = 5.0
wL = 1.0
csp = 1.0
rpi_params = RPIParams([:Hf, :O], n_body, max_deg, wL, csp, r0, rcutoff)

# Define auxiliary functions to assemble the matrix A
calc_B(systems) = vcat((evaluate_basis.(systems, [rpi_params])'...))
calc_dB(systems) =
    #vcat([vcat(d...) for d in ThreadsX.collect(evaluate_basis_d(s, rpi_params) for s in systems)]...)
    vcat([vcat(d...) for d in evaluate_basis_d.(systems, [rpi_params])]...)
calc_F(forces) = vcat([vcat(vcat(f...)...) for f in forces]...)

# Calculate A matrix
B = @time calc_B(train_systems)
dB = @time calc_dB(train_systems)
A = [B; dB]

# Calculate b vector (energies and forces)
e = train_energies
f = calc_F(train_forces)
b = [e; f]

# Calculate coefficients β
Q = Diagonal([0.5 .+ 0.0 * e; 90.0 .+ 0.0*f])
β = (A'*Q*A) \ (A'*Q*b) 

print(β)
write("beta.dat", "$β")

# Compute testing errors
B = nothing
dB = nothing
A = nothing
e = nothing
f = nothing
b = nothing
Q = nothing
GC.gc()

B_test = calc_B(test_systems)
dB_test = calc_dB(test_systems)
e_test = test_energies
f_test = calc_F(test_forces)

e_pred = B_test * β
e_error = abs.(e_pred .- e_test) ./ abs.(e_test)
f_pred = dB_test * β
f_error = abs.(f_pred .- f_test) ./ abs.(f_test)
println(mean(e_error), ", " , mean(f_error))


write("error.dat", "$(e_pred), $(f_error)")

