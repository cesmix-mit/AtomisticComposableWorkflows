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
using Plots

include("load-data.jl")


# Input: experiment_path, dataset_path, dataset_file, n_body, max_deg, r0, rcutoff, wL, csp
if size(ARGS, 1) == 0
    input = ["fit-ahfo2-ace/", "data/", "a-Hfo2-300K-NVT.extxyz",
             "1400", "2", "3", "1", "5", "1", "1"]
else
    input = ARGS
end

experiment_path = input[1]
run(`mkdir -p $experiment_path`)


# Load dataset #################################################################
dataset_path = input[2]
dataset_filename = input[3]
systems, energies, forces, stresses = load_data(dataset_path*dataset_filename)

# Split into training and testing
n_systems = parse(Int64, input[4]) # length(systems)
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

# Linearize energies and forces
calc_F(forces) = vcat([vcat(vcat(f...)...) for f in forces]...)
e_train = train_energies
f_train = calc_F(train_forces)
e_test = test_energies
f_test = calc_F(test_forces)
write(experiment_path*"e_train.dat", "$(e_train)")
write(experiment_path*"f_train.dat", "$(f_train)")
write(experiment_path*"e_test.dat", "$(e_train)")
write(experiment_path*"f_test.dat", "$(f_train)")


# Define RPI parameters ########################################################
n_body = parse(Int64, input[5])
max_deg = parse(Int64, input[6])
r0 = parse(Float64, input[7])
rcutoff = parse(Float64, input[8])
wL = parse(Float64, input[9])
csp = parse(Float64, input[10])
rpi_params = RPIParams([:Hf, :O], n_body, max_deg, wL, csp, r0, rcutoff)
write(experiment_path*"rpi_params.dat", "$(rpi_params)")


# Calculate descriptors ########################################################
calc_B(systems) = vcat((evaluate_basis.(systems, [rpi_params])'...))
calc_dB(systems) =
    vcat([vcat(d...) for d in evaluate_basis_d.(systems, [rpi_params])]...)
    #vcat([vcat(d...) for d in ThreadsX.collect(evaluate_basis_d(s, rpi_params) for s in systems)]...)
B_time = @time @elapsed B_train = calc_B(train_systems)
dB_time = @time @elapsed dB_train = calc_dB(train_systems)
B_test = calc_B(test_systems)
dB_test = calc_dB(test_systems)
write(experiment_path*"B_train.dat", "$(B_train)")
write(experiment_path*"dB_train.dat", "$(dB_train)")
write(experiment_path*"B_test.dat", "$(B_test)")
write(experiment_path*"dB_test.dat", "$(dB_test)")


# Calculate A and b ############################################################
A = [B_train; dB_train]
b_train = [e_train; f_train]


# Calculate coefficients β #####################################################
Q = Diagonal([0.5 .+ 0.0 * e_train; 90.0 .+ 0.0*f_train])
time_train = @time @elapsed  β = (A'*Q*A) \ (A'*Q*b_train)
n_params = size(β,1)
write(experiment_path*"beta.dat", "$β")


# Compute errors ##############################################################
function compute_errors(x_pred, x)
    x_rmse = sqrt(sum((x_pred .- x).^2) / length(x))
    x_mae = sum(abs.(x_pred .- x)) / length(x)
    x_mre = mean(abs.((x_pred .- x) ./ x))
    x_maxre = maximum(abs.((x_pred .- x) ./ x))
    return x_rmse, x_mae, x_mre, x_maxre
end

# Compute predictions 
e_train_pred = B_train * β; f_train_pred = dB_train * β
e_test_pred = B_test * β; f_test_pred = dB_test * β

# Compute errors
e_train_rmse, e_train_mae, e_train_mre, e_train_maxre = compute_errors(e_train_pred, e_train)
f_train_rmse, f_train_mae, f_train_mre, f_train_maxre = compute_errors(f_train_pred, f_train)
e_test_rmse, e_test_mae, e_test_mre, e_test_maxre = compute_errors(e_test_pred, e_test)
f_test_rmse, f_test_mae, f_test_mre, f_test_maxre = compute_errors(f_test_pred, f_test)


# Save results #################################################################
write(experiment_path*"results.csv", "dataset,\
                      n_systems,n_params,n_body,max_deg,r0,rcutoff,wL,csp,\
                      e_train_rmse,e_train_mae,e_train_mre,e_train_maxre,\
                      f_train_rmse,f_train_mae,f_train_mre,f_train_maxre,\
                      e_test_rmse,e_test_mae,e_test_mre,e_test_maxre,\
                      f_test_rmse,f_test_mae,f_test_mre,f_test_maxre,\
                      B_time,dB_time,time_train
                      $(dataset_filename), \
                      $(n_systems),$(n_params),$(n_body),$(max_deg),$(r0),$(rcutoff),$(wL),$(csp),\
                      $(e_train_rmse),$(e_train_mae),$(e_train_mre),$(e_train_maxre),\
                      $(f_train_rmse),$(f_train_mae),$(f_train_mre),$(f_train_maxre),\
                      $(e_test_rmse),$(e_test_mae),$(e_test_mre),$(e_test_maxre),\
                      $(f_test_rmse),$(f_test_mae),$(f_test_mre),$(f_test_maxre),\
                      $(B_time),$(dB_time),$(time_train)")

write(experiment_path*"results-short.csv", "dataset,\
                      n_systems,n_params,n_body,max_deg,r0,rcutoff,\
                      e_test_rmse,e_test_mae,\
                      f_test_rmse,f_test_mae,\
                      B_time,dB_time,time_train
                      $(dataset_filename),\
                      $(n_systems),$(n_params),$(n_body),$(max_deg),$(r0),$(rcutoff),\
                      $(e_test_rmse),$(e_test_mae),\
                      $(f_test_rmse),$(f_test_mae),\
                      $(B_time),$(dB_time),$(time_train)")

e = plot( e_test, e_test_pred, seriestype = :scatter, markerstrokewidth=0,
          label="", xlabel = "E DFT | eV/atom", ylabel = "E predicted | eV/atom")
savefig(e, experiment_path*"e.png")

f = plot( f_test, f_test_pred, seriestype = :scatter, markerstrokewidth=0,
          label="", xlabel = "F DFT | eV/Å", ylabel = "F predicted | eV/Å")
savefig(f, experiment_path*"f.png")



