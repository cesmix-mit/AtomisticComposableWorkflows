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

# Load training and test datasets ##############################################

filename = ARGS[1] # "a-Hfo2-300K-NVT.extxyz" # "data/HfO2_relax_1000.xyz"
systems, energies, forces, stresses = load_data("../data/"*filename)

# Split into training, testing
n_systems = parse(Int64, ARGS[2]) # length(systems)
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

# Create RPI Basis #############################################################

n_body = parse(Int64, ARGS[3])
max_deg = parse(Int64, ARGS[4])
r0 = parse(Float64, ARGS[5])
rcutoff = parse(Float64, ARGS[6])
wL = parse(Float64, ARGS[7])
csp = parse(Float64, ARGS[8])
rpi_params = RPIParams([:Hf, :O], n_body, max_deg, wL, csp, r0, rcutoff)

# Define auxiliary functions to assemble the matrix A
calc_B(systems) = vcat((evaluate_basis.(systems, [rpi_params])'...))
calc_dB(systems) =
    #vcat([vcat(d...) for d in ThreadsX.collect(evaluate_basis_d(s, rpi_params) for s in systems)]...)
    vcat([vcat(d...) for d in evaluate_basis_d.(systems, [rpi_params])]...)

calc_F(forces) = vcat([vcat(vcat(f...)...) for f in forces]...)

# Calculate A matrix ###########################################################
B_time = @time @elapsed B = calc_B(train_systems)
dB_time = @time @elapsed dB = calc_dB(train_systems)

A = [B; dB]

write("A.dat", "$A")

# Calculate b vector (energies and forces) #####################################
e = train_energies
f = calc_F(train_forces)
b = [e; f]

write("b.dat", "$b")

# Calculate coefficients β #####################################################
Q = Diagonal([0.5 .+ 0.0 * e; 90.0 .+ 0.0*f])
β = (A'*Q*A) \ (A'*Q*b)

write("beta.dat", "$β")

# Compute testing errors #######################################################

B = dB = A = e = f = b = Q = nothing; GC.gc()

B_test = calc_B(test_systems)
dB_test = calc_dB(test_systems)
e_test = test_energies
f_test = calc_F(test_forces)

e_pred = B_test * β
f_pred = dB_test * β

e_max_rel_error = maximum(abs.((e_pred .- e_test) ./ e_test))
f_max_rel_error = maximum(abs.((f_pred .- f_test) ./ f_test))

e_mean_rel_error = mean(abs.((e_pred .- e_test) ./ e_test))
f_mean_rel_error = mean(abs.((f_pred .- f_test) ./ f_test))

e_mean_abs_error = mean(abs.(e_pred .- e_test) ./ length(e_test))
f_mean_abs_error = mean(abs.(f_pred .- f_test) ./ length(f_test))

e_rmse = sqrt(sum((e_pred .- e_test).^2) / length(e_test))
f_rmse = sqrt(sum((f_pred .- f_test).^2) / length(f_test))

write("result.dat", "$(filename), \
                     $(e_max_rel_error), $(e_mean_rel_error), $(e_mean_abs_error), $(e_rmse), \
                     $(f_max_rel_error), $(f_mean_rel_error), $(f_mean_abs_error), $(f_rmse), \
                     $(n_systems), $(length(β)), \
                     $(n_body), $(max_deg), $(r0), $(rcutoff), $(wL), $(csp), \
                     $(B_time), $(dB_time)")




