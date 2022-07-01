# This code will be used to enrich InteratomicPotentials.jl, 
# InteratomicBasisPotentials.jl, and PotentialLearning.jl.

using AtomsBase
using InteratomicPotentials 
using InteratomicBasisPotentials
using LinearAlgebra 
#using MKL
using Random
using StaticArrays
using Statistics 
using StatsBase
using Optimization
using OptimizationOptimJL
using ForwardDiff
using UnitfulAtomic
using Unitful 
using Flux
using Flux.Data: DataLoader
using Zygote
using ForwardDiff
using CUDA
using BenchmarkTools
using Plots

include("input.jl")
include("NNBasisPotential.jl")
include("training.jl")
include("postproc.jl")
include("utils.jl")


# Load input parameters
input = get_input()


# Create experiment folder
path = "neural-ace-"*input["experiment_path"]
run(`mkdir -p $path`)
@savevar path input


# Load dataset
train_systems, train_energies, train_forces, train_stresses,
test_systems, test_energies, test_forces, test_stresses = load_dataset(input)


# Linearize energies and forces
e_train, f_train, e_test, f_test =
        linearize(train_systems, train_energies, train_forces, train_stresses,
                  test_systems, test_energies, test_forces, test_stresses)
@savevar path e_train
@savevar path f_train
@savevar path e_test
@savevar path f_test


# Define ACE parameters
n_body = input["n_body"]
max_deg = input["max_deg"]
r0 = input["r0"]
rcutoff = input["rcutoff"]
wL = input["wL"]
csp = input["csp"]
atomic_symbols = unique(atomic_symbol(train_systems[1]))
ibp_params = ACEParams(atomic_symbols, n_body, max_deg, wL, csp, r0, rcutoff)
@savevar path ibp_params


# Calculate descriptors. TODO: add this to InteratomicBasisPotentials.jl?
calc_B(sys) = evaluate_basis.(sys, [ibp_params])
calc_dB(sys) = [ dBs_comp for dBs_sys in evaluate_basis_d.(sys, [ibp_params])
                          for dBs_atom in dBs_sys
                          for dBs_comp in eachrow(dBs_atom)]
B_time = @time @elapsed B_train = calc_B(train_systems)
dB_time = @time @elapsed dB_train = calc_dB(train_systems)
B_test = calc_B(test_systems)
dB_test = calc_dB(test_systems)
B_train_ext = vcat([ fill(B_train[i], 3length(position(s)))
                     for (i,s) in enumerate(train_systems)]...)
B_test_ext = vcat([ fill(B_test[i], 3length(position(s)))
                    for (i,s) in enumerate(test_systems)]...)
@savevar path B_train
@savevar path dB_train
@savevar path B_test
@savevar path dB_test


# Define neural network model
n_desc = length(first(B_test))
nn = Chain(Dense(n_desc,8,Flux.relu), Dense(8,1))
nn_params = Flux.params(nn)
n_params = sum(length, Flux.params(nn))
nnbp = NNBasisPotential(nn, nn_params, ibp_params)


# Define batches
train_loader_e, train_loader_f, test_loader_e, test_loader_f = 
               get_batches(B_train, B_train_ext, e_train, dB_train, f_train,
                           B_test, B_test_ext, e_test, dB_test, f_test)


# Train
println("Training energies and forces...")
lib = "Optimization.jl"; epochs = 10; opt = BFGS(); maxiters = 30
w_e, w_f = input["w_e"], input["w_f"]
time_fitting =
@time @elapsed train_losses_epochs, test_losses_epochs, train_losses_batches = 
            train!( lib, nnbp, epochs, opt, maxiters, train_loader_e,
                    train_loader_f, test_loader_e, test_loader_f, w_e, w_f)

println("time_fitting:", time_fitting)
@savevar path train_losses_batches
@savevar path train_losses_epochs
@savevar path test_losses_epochs
@savevar path nnbp.nn_params


# Calculate predictions
e_train_pred = potential_energy.(B_train, [nnbp])
f_train_pred = force.(B_train_ext, dB_train, [nnbp])
e_test_pred = potential_energy.(B_test, [nnbp])
f_test_pred = force.(B_test_ext, dB_test, [nnbp])


# Post-process output: calculate metrics, save results and plots
postproc( input, e_train_pred, e_train, f_train_pred, f_train,
          e_test_pred, e_test, f_test_pred, f_test,
          n_params, B_time, dB_time, time_fitting)


