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
using Flux
using Flux.Data: DataLoader
using BSON: @save
using CUDA
using BenchmarkTools

include("load_data.jl")


# Input: dataset_path, dataset_filename, n_body, max_deg, r0, rcutoff, wL, csp
ARGS = ["data/", "a-Hfo2-300K-NVT.extxyz", "1000", "2", "3", "1", "5", "1", "1"]


# Load training and test datasets ##############################################
dataset_path = ARGS[1]
dataset_filename = ARGS[2]
systems, energies, forces, stresses = load_data(dataset_path*dataset_filename)


# Split into training, testing
n_systems = parse(Int64, ARGS[3]) # length(systems)
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
n_body = parse(Int64, ARGS[4])
max_deg = parse(Int64, ARGS[5])
r0 = parse(Float64, ARGS[6])
rcutoff = parse(Float64, ARGS[7])
wL = parse(Float64, ARGS[8])
csp = parse(Float64, ARGS[9])
rpi_params = RPIParams([:Hf, :O], n_body, max_deg, wL, csp, r0, rcutoff)


# Calculate descriptors ########################################################
calc_B(sys) = evaluate_basis.(sys, [rpi_params])
calc_dB(sys) = [ dBs_comp for dBs_sys in evaluate_basis_d.(sys, [rpi_params])
                          for dBs_atom in dBs_sys
                          for dBs_comp in eachrow(dBs_atom)]
B_time = @time @elapsed B_train = calc_B(train_systems)
dB_time = @time @elapsed dB_train = calc_dB(train_systems)
#write("B_train.dat", "$(B_train)")
#write("dB_train.dat", "$(dB_train)")


# Calculate train energies and forces) #########################################
e_train = train_energies
f_train = vcat([vcat(vcat(f...)...) for f in train_forces]...)
#write("e_train.dat", "$(e_train)")
#write("f_train.dat", "$(f_train)")


# Calculate neural network parameters ##########################################
train_loader = DataLoader(([B_train; dB_train] , [e_train; f_train]),
                            batchsize=128, shuffle=true)
n_desc = size(B_train[1], 1)
model = Chain(Dense(n_desc,100,Flux.relu),Dense(100,1))
nn(d) = sum(model(d))
ps = Flux.params(model)
n_params = sum(length, Flux.params(model))
loss(b_pred, b) = mean(abs.((b_pred .- b) ./ b))
global_loss(loader) =
    sum([loss(nn.(d), b) for (d, b) in loader]) / length(loader)
opt = ADAM(0.0001)
epochs = 5
for epoch in 1:epochs
    # Training of one epoch
    time = Base.@elapsed for (d, b) in train_loader
        gs = gradient(() -> loss(nn.(d), b), ps)
        Flux.Optimise.update!(opt, ps, gs)
    end
    # Report traning loss
    println("Epoch: $(epoch), loss: $(global_loss(train_loader)), time: $(time)")
end


# Compute errors ##############################################################
function compute_errors(x_pred, x)
    x_rmse = sqrt(sum((x_pred .- x).^2) / length(x))
    x_mae = mean(abs.(x_pred .- x) ./ length(x))
    x_mre = mean(abs.((x_pred .- x) ./ x))
    x_maxre = maximum(abs.((x_pred .- x) ./ x))
    return x_rmse, x_mae, x_mre, x_maxre
end

# Compute training errors
e_train_pred = nn.(B_train)
f_train_pred = nn.(dB_train)
e_train_rmse, e_train_mae, e_train_mre, e_train_maxre = compute_errors(e_train_pred, e_train)
f_train_rmse, f_train_mae, f_train_mre, f_train_maxre = compute_errors(f_train_pred, f_train)

# Compute test errors
B_test = calc_B(test_systems)
dB_test = calc_dB(test_systems)
e_test = test_energies
f_test = vcat([vcat(vcat(f...)...) for f in test_forces]...)
e_test_pred = nn.(B_test)
f_test_pred = nn.(dB_test)
e_test_rmse, e_test_mae, e_test_mre, e_test_maxre = compute_errors(e_test_pred, e_test)
f_test_rmse, f_test_mae, f_test_mre, f_test_maxre = compute_errors(f_test_pred, f_test)


## Save results #################################################################
write("results-nn.csv", "dataset,\
                      n_systems,n_params,n_body,max_deg,r0,rcutoff,wL,csp,\
                      e_train_rmse,e_train_mae,e_train_mre,e_train_maxre,\
                      f_train_rmse,f_train_mae,f_train_mre,f_train_maxre,\
                      e_test_rmse,e_test_mae,e_test_mre,e_test_maxre,\
                      f_test_rmse,f_test_mae,f_test_mre,f_test_maxre,\
                      B_time,dB_time
                      $(dataset_filename), \
                      $(n_systems),$(n_params),$(n_body),$(max_deg),$(r0),$(rcutoff),$(wL),$(csp),\
                      $(e_train_rmse),$(e_train_mae),$(e_train_mre),$(e_train_maxre),\
                      $(f_train_rmse),$(f_train_mae),$(f_train_mre),$(f_train_maxre),\
                      $(e_test_rmse),$(e_test_mae),$(e_test_mre),$(e_test_maxre),\
                      $(f_test_rmse),$(f_test_mae),$(f_test_mre),$(f_test_maxre),\
                      $(B_time),$(dB_time)")

