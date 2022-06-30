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
using BSON: @save
using CUDA
using BenchmarkTools
using Plots
#using Profile, FileIO

# Load input parameters ########################################################
# This section will feed user main script and/or PotentialLearning.jl
if size(ARGS, 1) == 0
    #input = ["fit-ahfo2-neural-ace/", "data/", "a-Hfo2-300K-NVT.extxyz",
    #         "100", "2", "3", "1", "5", "1", "1", "1", "1"]
    input = ["fit-ahfo2-neural-ace/", "data/", "a-Hfo2-300K-NVT.extxyz",
             "100", "3", "3", "1", "5", "1", "1", "3", "0.1"]
    #input = ["fit-TiO2-neural-ace/", "data/", "TiO2trainingset.xyz",
    #         "100", "3", "3", "1", "5", "1", "1", "1", "1"]
else
    input = ARGS
end
input = Dict( "experiment_path"     => input[1],
              "dataset_path"        => input[2],
              "dataset_filename"    => input[3],
              "n_systems"           => parse(Int64, input[4]),
              "n_body"              => parse(Int64, input[5]),
              "max_deg"             => parse(Int64, input[6]),
              "r0"                  => parse(Float64, input[7]),
              "rcutoff"             => parse(Float64, input[8]),
              "wL"                  => parse(Float64, input[9]),
              "csp"                 => parse(Float64, input[10]),
              "e_weight"            => parse(Float64, input[11]),
              "f_weight"            => parse(Float64, input[12]))


# Create experiment folder #####################################################
# This section will feed user main script and/or PotentialLearning.jl
experiment_path = input["experiment_path"]
run(`mkdir -p $experiment_path`)
write(experiment_path*"input.dat", "$input")


# Load dataset #################################################################
# This section will feed PotentialLearning.jl
include("load-data.jl")
filename = input["dataset_path"]*input["dataset_filename"]
systems, energies, forces, stresses = load_data(filename,
                                                max_entries = input["n_systems"])

# Split into training and testing
n_systems = length(systems)
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
f_test_v = vcat(test_forces...)
f_test = calc_F(test_forces)
write(experiment_path*"e_train.dat", "$(e_train)")
write(experiment_path*"f_train.dat", "$(f_train)")
write(experiment_path*"e_test.dat", "$(e_test)")
write(experiment_path*"f_test.dat", "$(f_test)")


# Define IBP parameters ########################################################
# This section will feed user main script
n_body = input["n_body"]
max_deg = input["max_deg"]
r0 = input["r0"]
rcutoff = input["rcutoff"]
wL = input["wL"]
csp = input["csp"]
atomic_symbols = unique(atomic_symbol(systems[1]))
ibp_params = ACEParams(atomic_symbols, n_body, max_deg, wL, csp, r0, rcutoff)
write(experiment_path*"ibp_params.dat", "$(ibp_params)")


# Calculate descriptors ########################################################
# This section will feed PotentialLearning.jl
calc_B(sys) = evaluate_basis.(sys, [ibp_params])
calc_dB(sys) = [ dBs_comp for dBs_sys in evaluate_basis_d.(sys, [ibp_params])
                          for dBs_atom in dBs_sys
                          for dBs_comp in eachrow(dBs_atom)]
B_time = @time @elapsed B_train = calc_B(train_systems)
dB_time = @time @elapsed dB_train = calc_dB(train_systems)
B_test = calc_B(test_systems)
dB_test = calc_dB(test_systems)
B_train_ext = vcat([ fill(B_train[i], 3length(position(s))) for (i,s) in enumerate(train_systems)]...)
B_test_ext = vcat([ fill(B_test[i], 3length(position(s))) for (i,s) in enumerate(test_systems)]...)
write(experiment_path*"B_train.dat", "$(B_train)")
write(experiment_path*"dB_train.dat", "$(dB_train)")
write(experiment_path*"B_test.dat", "$(B_test)")
write(experiment_path*"dB_test.dat", "$(dB_test)")

time_fitting = Base.@elapsed begin

# Define training and testing data #############################################
# This section will feed PotentialLearning.jl

# Normalize and split data into batches
e_ref = 1 #maximum(abs.(e_train))
f_ref = 1 #maximum(abs.(f_train))
B_ref = 1 #maximum([maximum(abs.(b)) for b in B_train])
dB_ref = 1 #1/B_ref

bs_train_e = floor(Int, length(B_train) * 0.125 ) # 0.5, 0.025
train_loader_e   = DataLoader((B_train / B_ref, e_train / e_ref),
                               batchsize=bs_train_e, shuffle=true)
bs_train_f = floor(Int, length(dB_train) * 0.125) # 0.025
train_loader_f   = DataLoader((B_train_ext / B_ref,
                               dB_train / dB_ref,
                               f_train / f_ref),
                               batchsize=bs_train_f, shuffle=true)
println("batchsize_e:", bs_train_e, ", batchsize_f:", bs_train_f)


bs_test_e = floor(Int, length(B_test) * 0.05)
test_loader_e   = DataLoader((B_test / B_ref, e_test / e_ref),
                              batchsize=bs_test_e, shuffle=true)
bs_test_f = floor(Int, length(dB_test) * 0.05)
test_loader_f   = DataLoader((B_test_ext / B_ref,
                              dB_test / dB_ref,
                              f_test / f_ref),
                              batchsize=bs_test_f, shuffle=true)
println("batchsize_e:", bs_test_e, ", batchsize_f:", bs_test_f)

# Define neural network model ##################################################
# This section will feed InteratomicPotentials.jl/InteratomicBasisPotentials.jl

# Defining NNBP composed type and associated functions
mutable struct NNBasisPotential <: AbstractPotential
    nn
    nn_params
    ibp_params
end

function potential_energy(A::AbstractSystem, 
                          p::NNBasisPotential)
    b = evaluate_basis(A, p.ibp_params)
    return p.nn(b)
end

function force(A::AbstractSystem, 
               p::NNBasisPotential)
    b = evaluate_basis(A, p.ibp_params)
    dnndb = first(gradient(p.nn, b))
    dbdr = evaluate_basis_d(A, p.ibp_params)
    return [[-dnndb ⋅ dbdr[atom][coor,:] for coor in 1:3]
             for atom in 1:length(dbdr)]
end

function potential_energy(b::Vector, p::NNBasisPotential)
    return sum(p.nn(b))
end

function potential_energy(b::Vector, ps::Vector, re)
    return sum(re(ps)(b))
end

# Note: calculating the gradient of the loss function requires in turn
# calculating the gradient of the energy. That is, calculating the gradient of
# a function that calculates another gradient.
# So far I have not found a clean way to do this using the abstractions 
# provided by Flux, which in turn is integrated with Zygote. 
# Links related to this issue:
#    https://discourse.julialang.org/t/how-to-add-norm-of-gradient-to-a-loss-function/69873/16
#    https://discourse.julialang.org/t/issue-with-zygote-over-forwarddiff-derivative/70824
#    https://github.com/FluxML/Zygote.jl/issues/953#issuecomment-841882071
#
# To solve this for the moment I am calculating one of the gradients analytically.
# To do this I had to use Flux.destructure, which I think makes the code slower
# because of the copies it creates.

function grad_mlp(nn_params, x0)
    dsdy(x) = x>0 ? 1 : 0 # Flux.σ(x) * (1 - Flux.σ(x)) 
    prod = 1; x = x0
    n_layers = length(nn_params) ÷ 2
    for i in 1:2:2(n_layers-1)-1  # i : 1, 3
        y = nn_params[i] * x + nn_params[i+1]
        x =  Flux.relu.(y) # Flux.σ.(y)
        prod = dsdy.(y) .* nn_params[i] * prod
    end
    i = 2(n_layers)-1 
    prod = nn_params[i] * prod
    return prod
end

function force(b, dbdr, p)
    dnndb = grad_mlp(p.nn_params, b)
    return dnndb ⋅ dbdr
end

function force(b, dbdr, ps, re)
    nn_params = Flux.params(re(ps))
    dnndb = grad_mlp(nn_params, b)
    return dnndb ⋅ dbdr
end

# Computing the force using ForwardDiff
#function force(b::Vector, dbdr::Vector, p::NNBasisPotential)
#    dnndb = ForwardDiff.gradient(x -> sum(p.nn(x)), b)
#    return dnndb ⋅ dbdr
#end

#function force(b::Vector, dbdr::Vector, p::NNBasisPotential)
#    dnndb = gradient(x -> sum(p.nn(x)), b)[1]
#    return dnndb ⋅ dbdr
#end

# Computing the force using ForwardDiff and destructure
#function force(b::Vector, dbdr::Vector, ps::Vector, re)
#    dnndb = ForwardDiff.gradient(x -> sum(re(ps)(x)), b)
#    return dnndb ⋅ dbdr
#end

# Computing the force using pullback
#function force(b::Vector, dbdr::Vector, p::NNBasisPotential)
#    y, pullback = Zygote.pullback(p.nn, b)
#    dnndb = pullback(ones(size(y)))[1]
#    return dnndb ⋅ dbdr
#end

#function force(b::Vector, dbdr::Vector,  ps::Vector, re)
#    y, pullback = Zygote.pullback(re(ps), b)
#    dnndb = pullback(ones(size(y)))[1]
#    return dnndb ⋅ dbdr
#end

# Define neural network model
n_desc = length(first(train_loader_e)[1][1])
nn = Chain(Dense(n_desc,8,Flux.relu), Dense(8,1))
nn_params = Flux.params(nn)
n_params = sum(length, Flux.params(nn))

# Define neural network basis potential
nnbp = NNBasisPotential(nn, nn_params, ibp_params)

end

# Train ########################################################################
# This section will feed PotentialLearning.jl

println("Training energies and forces...")

# Define loss functions
w_e = input["e_weight"]; w_f = input["f_weight"]
loss(es_pred, es, fs_pred, fs) =  w_e * Flux.Losses.mse(es_pred, es) +
                                  w_f * Flux.Losses.mse(fs_pred, fs)
global_loss(loader_e, loader_f, ps, re) =
    mean([loss(potential_energy.(bs_e, [ps], [re]), es, force.(bs_f, dbs_f, [ps], [re]), fs)
          for ((bs_e, es), (bs_f, dbs_f, fs)) in zip(loader_e, loader_f)])

training_losses = []
testing_losses = []
batch_train_losses = []

# Training using ADAM from Flux.jl
function train_adam()
    epochs = 100
    opt = ADAM(0.001) # opt = ADAM(0.002, (0.9, 0.999)) 
    ps, re = Flux.destructure(nnbp.nn)
    for epoch in 1:epochs
        global time_fitting += Base.@elapsed for ((bs_e, es), (bs_f, dbs_f, fs)) in
                                                  zip(train_loader_e, train_loader_f)
            g = gradient(Flux.params(ps)) do
                loss(potential_energy.(bs_e, [ps], [re]), es,
                     force.(bs_f, dbs_f, [ps], [re]), fs)
            end
            Flux.Optimise.update!(opt, Flux.params(ps), g)
        end
        
        # Report losses
        training_loss = mean([loss(potential_energy.(bs_e, [ps], [re]), es, 
                                   force.(bs_f, dbs_f, [ps], [re]), fs)
                              for ((bs_e, es), (bs_f, dbs_f, fs)) in
                                  zip(train_loader_e, train_loader_f)])
        testing_loss = mean([loss(potential_energy.(bs_e, [ps], [re]), es, 
                                  force.(bs_f, dbs_f, [ps], [re]), fs)
                              for ((bs_e, es), (bs_f, dbs_f, fs)) in
                                  zip(test_loader_e, test_loader_f)])
        push!(training_losses, training_loss)
        push!(testing_losses, testing_loss)
        println("Epoch $(epoch). Losses of complete datasets: \
                 training loss: $(training_loss), \
                 testing loss: $(testing_loss).")
    end
    nnbp.nn = re(ps)
    nnbp.nn_params = Flux.params(nnbp.nn)
end

# Training using BFGS from Optimization.jl
# I am using one of the Optimization.jl solvers, because I have not been able to
# tackle this problem with the Flux solvers.
function train_bfgs()
    epochs = 2
    ps, re = Flux.destructure(nnbp.nn)
    for epoch in 1:epochs
        # Train through batches
        i = 1
        global time_fitting += Base.@elapsed for ((bs_e, es), (bs_f, dbs_f, fs)) in
                                                  zip(train_loader_e, train_loader_f)
            batch_loss(ps, p) = loss(potential_energy.(bs_e, [ps], [re]), es, 
                                     force.(bs_f, dbs_f, [ps], [re]), fs)
            dbatchlossdps = OptimizationFunction(batch_loss,
                                                 Optimization.AutoForwardDiff()) # Optimization.AutoZygote()
            prob = OptimizationProblem(dbatchlossdps, ps, []) # prob = remake(prob,u0=sol.minimizer)
            callback = function (p, l)
                println("Epoch: $(epoch), batch: $i, training loss: $l")
                push!(batch_train_losses, l)
                return false
            end
            sol = solve(prob, BFGS(), callback=callback, maxiters=30) # reltol = 1e-14
            ps = sol.u
            i = i + 1
        end
        
        # Report losses
        training_loss = mean([loss(potential_energy.(bs_e, [ps], [re]), es, 
                                   force.(bs_f, dbs_f, [ps], [re]), fs)
                              for ((bs_e, es), (bs_f, dbs_f, fs)) in
                                  zip(train_loader_e, train_loader_f)])
        testing_loss = mean([loss(potential_energy.(bs_e, [ps], [re]), es, 
                                  force.(bs_f, dbs_f, [ps], [re]), fs)
                              for ((bs_e, es), (bs_f, dbs_f, fs)) in
                                  zip(test_loader_e, test_loader_f)])
        push!(training_losses, training_loss)
        push!(testing_losses, testing_loss)
        println("Epoch $(epoch). Losses of complete datasets: \
                 training loss: $(training_loss), \
                 testing loss: $(testing_loss).")
    end
    nnbp.nn = re(ps)
    nnbp.nn_params = Flux.params(nnbp.nn)
end

train_bfgs()

# Training using BFGS from Optimizer.jl (1 batch)
#time_fitting += Base.@elapsed begin
#    ps, re = Flux.destructure(nnbp.nn)
#    ((bs_e, es), (bs_f, dbs_f, fs)) = collect(zip(train_loader_e, train_loader_f))[1]
#    loss(ps, p) = loss(potential_energy.(bs_e, [ps], [re]), es, 
#                       force.(bs_f, dbs_f, [ps], [re]), fs)
#    dlossdps = OptimizationFunction(loss, Optimization.AutoForwardDiff()) # Optimization.AutoZygote()
#    prob = OptimizationProblem(dlossdps, ps, []) #prob = remake(prob,u0=sol.minimizer)
#    callback = function (p, l)
#        println("Thread: $(Threads.threadid()) current loss is: $l")
#        return false
#    end
#    sol = solve(prob, BFGS(), callback=callback, maxiters=1000) # reltol = 1e-14
#    ps = sol.u
#    nnbp.nn = re(ps)
#    nnbp.nn_params = Flux.params(nnbp.nn)
#end

#-------------------------------------------------------------------------------
## Training using BFGS and data parallelism with Base.Threads
# execute: julia --threads 4 fit-ahfo2-neural-ace.jl
# BLAS.set_num_threads(1)
##time_fitting += Base.@elapsed begin
#ps, re = Flux.destructure(nnbp.nn)
#nt = Threads.nthreads() 
#loaders = collect(zip(train_loader_e, train_loader_f))[1:nt]
## Compute loss of each batch
#opt_func = Array{Function}(undef, nt);
#for tid in 1:nt
#    ((bs_e, es), (bs_f, dbs_f, fs)) = loaders[tid]
#    batch_loss(ps, p) = loss(potential_energy.(bs_e, [ps], [re]), es, 
#                             force.(bs_f, dbs_f, [ps], [re]), fs)
#    opt_func[tid] = batch_loss
#end
## Compute total loss and define optimization function
#total_loss(ps, p) = mean([f(ps, p) for f in opt_func])
#dlossdps = OptimizationFunction(total_loss, Optimization.AutoForwardDiff()) # Optimization.AutoZygote())
## Optimize using averaged gradient on each batch
#callback = function (p, l)
#    println("Thread $(Threads.threadid()), current loss is: $l")
#    return false
#end
#pss = [deepcopy(ps) for i in 1:nt]
##@profile Threads.@threads for tid in 1:nt
#    ((bs_e, es), (bs_f, dbs_f, fs)) = loaders[tid]
#    ps_i = deepcopy(ps)
#    prob = OptimizationProblem(dlossdps, ps_i, []) # prob = remake(prob,u0=sol.minimizer)
#    sol = solve(prob, BFGS(), callback = callback, maxiters=10_000_000)
#    pss[tid] = sol.u
#end
## Average parameters
#ps = mean(pss)
#nnbp.nn = re(ps)
#nnbp.nn_params = Flux.params(nnbp.nn)
#end
#save("fit-ahfo2-neural-ace.jlprof",  Profile.retrieve()...)
#using ProfileView, FileIO
#data = load("fit-ahfo2-neural-ace.jlprof")
#ProfileView.view(data[1], lidict=data[2])
#-------------------------------------------------------------------------------

println("time_fitting:", time_fitting)

write(experiment_path*"batch_train_losses.dat", "$(batch_train_losses)")
write(experiment_path*"training_losses.dat", "$(training_losses)")
write(experiment_path*"testing_losses.dat", "$(testing_losses)")
write(experiment_path*"params.dat", "$(nnbp.nn_params)")


# Calculate predictions ########################################################
# This section will feed PotentialLearning.jl

e_train_pred = potential_energy.(B_train / B_ref, [nnbp]) * e_ref
f_train_pred = force.(B_train_ext / B_ref, dB_train / dB_ref, [nnbp]) * f_ref
e_test_pred = potential_energy.(B_test / B_ref, [nnbp]) * e_ref
f_test_pred = force.(B_test_ext / B_ref, dB_test / dB_ref, [nnbp]) * f_ref
f_test_pred_v = collect(eachcol(reshape(f_test_pred, 3, :)))


# Calculate metrics ############################################################
# This section will feed PotentialLearning.jl

function calc_metrics(x_pred, x)
    x_mae = sum(abs.(x_pred .- x)) / length(x)
    x_rmse = sqrt(sum((x_pred .- x).^2) / length(x))
    x_rsq = 1 - sum((x_pred .- x).^2) / sum((x .- mean(x)).^2)
    return x_mae, x_rmse, x_rsq
end

e_train_mae, e_train_rmse, e_train_rsq = calc_metrics(e_train_pred, e_train)
f_train_mae, f_train_rmse, f_train_rsq = calc_metrics(f_train_pred, f_train)
e_test_mae, e_test_rmse, e_test_rsq = calc_metrics(e_test_pred, e_test)
f_test_mae, f_test_rmse, f_test_rsq = calc_metrics(f_test_pred, f_test)

f_test_cos = dot.(f_test_v, f_test_pred_v) ./ (norm.(f_test_v) .* norm.(f_test_pred_v))
f_test_mean_cos = mean(f_test_cos)


# Save results #################################################################
# This section will feed user main script and/or PotentialLearning.jl

dataset_filename = input["dataset_filename"]
write(experiment_path*"results.csv", "dataset,\
                      n_systems,n_params,n_body,max_deg,r0,\
                      rcutoff,wL,csp,w_e,w_f,\
                      e_train_mae,e_train_rmse,e_train_rsq,\
                      f_train_mae,f_train_rmse,f_train_rsq,\
                      e_test_mae,e_test_rmse,e_test_rsq,\
                      f_test_mae,f_test_rmse,f_test_rsq,\
                      f_test_mean_cos,B_time,dB_time,time_fitting
                      $(dataset_filename), \
                      $(n_systems),$(n_params),$(n_body),$(max_deg),$(r0),\
                      $(rcutoff),$(wL),$(csp),$(w_e),$(w_e),\
                      $(e_train_mae),$(e_train_rmse),$(e_train_rsq),\
                      $(f_train_mae),$(f_train_rmse),$(f_train_rsq),\
                      $(e_test_mae),$(e_test_rmse),$(e_test_rsq),\
                      $(f_test_mae),$(f_test_rmse),$(f_test_rsq),\
                      $(f_test_mean_cos),$(B_time),$(dB_time),$(time_fitting)")

write(experiment_path*"results-short.csv", "dataset,\
                      n_systems,n_params,n_body,max_deg,r0,rcutoff,\
                      e_test_mae,e_test_rmse,\
                      f_test_mae,f_test_rmse,\
                      f_test_mean_cos,\
                      B_time,dB_time,time_fitting
                      $(dataset_filename),\
                      $(n_systems),$(n_params),$(n_body),$(max_deg),$(r0),$(rcutoff),\
                      $(e_test_mae),$(e_test_rmse),\
                      $(f_test_mae),$(f_test_rmse),\
                      $(B_time),$(dB_time),$(time_fitting)")

r0 = minimum(e_test); r1 = maximum(e_test); rs = (r1-r0)/10
plot( e_test, e_test_pred, seriestype = :scatter, markerstrokewidth=0,
      label="", xlabel = "E DFT | eV/atom", ylabel = "E predicted | eV/atom")
plot!( r0:rs:r1, r0:rs:r1, label="")
savefig(experiment_path*"e_test.png")

r0 = 0; r1 = ceil(maximum(norm.(f_test_v)))
plot( norm.(f_test_v), norm.(f_test_pred_v), seriestype = :scatter, markerstrokewidth=0,
      label="", xlabel = "|F| DFT | eV/Å", ylabel = "|F| predicted | eV/Å", 
      xlims = (r0, r1), ylims = (r0, r1))
plot!( r0:r1, r0:r1, label="")
savefig(experiment_path*"f_test.png")

plot( f_test_cos, seriestype = :scatter, markerstrokewidth=0,
      label="", xlabel = "F DFT vs F predicted", ylabel = "cos(α)")
savefig(experiment_path*"f_test_cos.png")

