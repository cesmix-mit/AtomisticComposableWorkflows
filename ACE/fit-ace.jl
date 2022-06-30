# This code will be used to enrich InteratomicPotentials.jl, 
# InteratomicBasisPotentials.jl, and PotentialLearning.jl.

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

# Load input parameters ########################################################
# This section will feed user main script and/or PotentialLearning.jl
if size(ARGS, 1) == 0
    # Define default input
#    input = [ "100", "2", "3", "1", "5", "1", "1", "1", "1",
#              "fit-ahfo2-neural-ace/", "data/", "a-Hfo2-300K-NVT.extxyz"]
#    input = [ "100", "3", "3", "1", "5", "1", "1", "3", "0.1",
#              "fit-ahfo2-neural-ace/", "data/", "a-Hfo2-300K-NVT.extxyz"]
    input = [ "100", "3", "3", "1", "5", "1", "1", "1", "1",
              "fit-TiO2-neural-ace-2/", "data/", 
              "TiO2trainingset.xyz", "TiO2testset.xyz",]
else
    # Define input from ARGS
    input = ARGS
end
input1 = Dict( "n_systems"           => parse(Int64, input[1]),
               "n_body"              => parse(Int64, input[2]),
               "max_deg"             => parse(Int64, input[3]),
               "r0"                  => parse(Float64, input[4]),
               "rcutoff"             => parse(Float64, input[5]),
               "wL"                  => parse(Float64, input[6]),
               "csp"                 => parse(Float64, input[7]),
               "e_weight"            => parse(Float64, input[8]),
               "f_weight"            => parse(Float64, input[9]))
if length(input[10:end]) == 3
    input2 = Dict( "experiment_path"     => input[10],
                   "dataset_path"        => input[11],
                   "dataset_filename"    => input[12])
else
    input2 = Dict( "experiment_path"       => input[10],
                   "dataset_path"          => input[11],
                   "trainingset_filename"  => input[12],
                   "testset_filename"      => input[13])
end
input = merge(input1, input2)


# Create experiment folder #####################################################
# This section will feed user main script and/or PotentialLearning.jl
experiment_path = input["experiment_path"]
run(`mkdir -p $experiment_path`)
write(experiment_path*"input.dat", "$input")


# Load dataset #################################################################
# This section will feed PotentialLearning.jl
include("load-data.jl")

if "dataset_filename" in keys(input)
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
else
    filename = input["dataset_path"]*input["trainingset_filename"]
    train_systems, train_energies, train_forces, train_stresses =
            load_data(filename, max_entries = input["n_systems"])
    filename = input["dataset_path"]*input["testset_filename"]
    test_systems, test_energies, test_forces, test_stresses =
            load_data(filename, max_entries = input["n_systems"])
end

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


# Define ACE parameters ########################################################
n_body = input["n_body"]
max_deg = input["max_deg"]
r0 = input["r0"]
rcutoff = input["rcutoff"]
wL = input["wL"]
csp = input["csp"]
atomic_symbols = unique(atomic_symbol(train_systems[1]))
ibp_params = ACEParams(atomic_symbols, n_body, max_deg, wL, csp, r0, rcutoff)
write(experiment_path*"ibp_params.dat", "$(ibp_params)")


# Calculate descriptors ########################################################
calc_B(sys) = vcat((evaluate_basis.(sys, [ibp_params])'...))
calc_dB(sys) =
    vcat([vcat(d...) for d in evaluate_basis_d.(sys, [ibp_params])]...)
    #vcat([vcat(d...) for d in ThreadsX.collect(evaluate_basis_d(s, ibp_params) for s in sys)]...)
B_time = @time @elapsed B_train = calc_B(train_systems)
dB_time = @time @elapsed dB_train = calc_dB(train_systems)
B_test = calc_B(test_systems)
dB_test = calc_dB(test_systems)
write(experiment_path*"B_train.dat", "$(B_train)")
write(experiment_path*"dB_train.dat", "$(dB_train)")
write(experiment_path*"B_test.dat", "$(B_test)")
write(experiment_path*"dB_test.dat", "$(dB_test)")


# Calculate A and b ############################################################
time_fitting = Base.@elapsed begin
A = [B_train; dB_train]
b = [e_train; f_train]

# Filter outliers
#fmean = mean(f_train); fstd = std(f_train)
#non_outliers = fmean - 2fstd .< f_train .< fmean + 2fstd 
#f_train = f_train[non_outliers]
#v = BitVector([ ones(length(e_train)); non_outliers])
#A = A[v , :]


# Calculate coefficients β #####################################################
e_weight = input["e_weight"]; f_weight = input["f_weight"]
Q = Diagonal([e_weight * ones(length(e_train));
              f_weight * ones(length(f_train))])
β = (A'*Q*A) \ (A'*Q*b)

end

## Check weights
#using IterTools
#for (e_weight, f_weight) in product(1:10:100, 1:10:100)
#    Q = Diagonal([e_weight * ones(length(e_train));
#                  f_weight * ones(length(f_train))])
#    try
#        β = (A'*Q*A) \ (A'*Q*b)
#        a = compute_errors(dB_test * β, f_test)
#        println(e_weight,", ", f_weight, ", ", a[1])
#    catch
#        println("Exception with :", e_weight,", ", f_weight)
#    end
#end

n_params = size(β,1)
write(experiment_path*"beta.dat", "$β")


# Calculate predictions ########################################################
e_train_pred = B_train * β
f_train_pred = dB_train * β
e_test_pred = B_test * β
f_test_pred = dB_test * β
f_test_pred_v = collect(eachcol(reshape(f_test_pred, 3, :)))


# Calculate metrics ############################################################
function calc_metrics(x_pred, x)
    x_mae = sum(abs.(x_pred .- x)) / length(x)
    x_mre = mean(abs.((x_pred .- x) ./ x))
    x_rmse = sqrt(sum((x_pred .- x).^2) / length(x))
    x_rsq = 1 - sum((x_pred .- x).^2) / sum((x .- mean(x)).^2)
    return x_mae, x_mre, x_rmse, x_rsq
end

e_train_mae, e_train_rmse, e_train_rsq = calc_metrics(e_train_pred, e_train)
f_train_mae, f_train_rmse, f_train_rsq = calc_metrics(f_train_pred, f_train)
e_test_mae, e_test_rmse, e_test_rsq = calc_metrics(e_test_pred, e_test)
f_test_mae, f_test_rmse, f_test_rsq = calc_metrics(f_test_pred, f_test)

f_test_cos = dot.(f_test_v, f_test_pred_v) ./ (norm.(f_test_v) .* norm.(f_test_pred_v))
f_test_mean_cos = mean(f_test_cos)


# Save results #################################################################
dataset_filename = input["dataset_filename"]
write(experiment_path*"results.csv", "dataset,\
                      n_systems,n_params,n_body,max_deg,r0,\
                      rcutoff,wL,csp,e_weight,f_weight,\
                      e_train_mae,e_train_rmse,e_train_rsq,\
                      f_train_mae,f_train_rmse,f_train_rsq,\
                      e_test_mae,e_test_rmse,e_test_rsq,\
                      f_test_mae,f_test_rmse,f_test_rsq,\
                      f_test_mean_cos,B_time,dB_time,time_fitting
                      $(dataset_filename), \
                      $(n_systems),$(n_params),$(n_body),$(max_deg),$(r0),\
                      $(rcutoff),$(wL),$(csp),$(e_weight),$(f_weight),\
                      $(e_train_mae),$(e_train_rmse),$(e_train_rsq),\
                      $(f_train_mae),$(f_train_rmse),$(f_train_rsq),\
                      $(e_test_mae),$(e_test_rmse),$(e_test_rsq),\
                      $(f_test_mae),$(f_test_rmse),$(f_test_rsq),\
                      $(f_test_mean_cos),$(B_time),$(dB_time),$(time_fitting)")

write(experiment_path*"results-short.csv", "dataset,\
                      n_systems,n_params,n_body,max_deg,r0,rcutoff,\
                      e_test_mae,e_test_rmse,\
                      f_test_mae,f_test_rmse,\
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

