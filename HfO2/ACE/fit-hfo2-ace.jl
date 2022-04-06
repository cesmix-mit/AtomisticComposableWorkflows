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

filename = ARGS[1]
systems, energies, forces, stresses = load_data("../../data/"*filename)

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
B_time = @time @elapsed B_train = calc_B(train_systems)
dB_time = @time @elapsed dB_train = calc_dB(train_systems)

A = [B_train; dB_train]

write("A.dat", "$A")

# Calculate b vector (energies and forces) #####################################
e_train = train_energies
f_train = calc_F(train_forces)
b_train = [e_train; f_train]

write("b.dat", "$(b_train)")

# Calculate coefficients β #####################################################
Q = Diagonal([0.5 .+ 0.0 * e; 90.0 .+ 0.0*f])
β = (A'*Q*A) \ (A'*Q*b_train)

write("beta.dat", "$β")




# Compute errors ##############################################################

function compute_errors(x_pred, x)
    x_rmse = sqrt(sum((x_pred .- x).^2) / length(x))
    x_mae = mean(abs.(x_pred .- x) ./ length(x))
    x_mre = mean(abs.((x_pred .- x) ./ x))
    x_maxre = maximum(abs.((x_pred .- x) ./ x))
    return x_rmse, x_mae, x_mre, x_maxre
end

# Compute training errors
e_train_pred = B_train * β
f_train_pred = dB_train * β
e_train_rmse, e_train_mae, e_train_mre, e_train_maxre = compute_errors(e_train_pred, e_train)
f_train_rmse, f_train_mae, f_train_mre, f_train_maxre = compute_errors(f_train_pred, f_train)

# Compute test errors
B_train = dB_train = A = b_train = Q = nothing; GC.gc()
B_test = calc_B(test_systems)
dB_test = calc_dB(test_systems)
e_test = test_energies
f_test = calc_F(test_forces)
e_test_pred = B_test * β
f_test_pred = dB_test * β
e_test_rmse, e_test_mae, e_test_mre, e_test_maxre = compute_errors(e_test_pred, e_test)
f_test_rmse, f_test_mae, f_test_mre, f_test_maxre = compute_errors(f_test_pred, f_test)


# Save results #################################################################
write("results.dat", "$(filename), \
                      $(n_systems),$(n_params,$(n_body,$(max_deg,$(r0,$(rcutoff),$(wL),$(csp),\
                      $(e_train_rmse),$(e_train_mae),$(e_train_mre),$(e_train_maxre),\
                      $(f_train_rmse),$(f_train_mae),$(f_train_mre),$(f_train_maxre),\
                      $(e_test_rmse),$(e_test_mae),$(e_test_mre),$(e_test_maxre),\
                      $(f_test_rmse),$(f_test_mae),$(f_test_mre),$(f_test_maxre),\
                      $(B_time),$(dB_time)")

