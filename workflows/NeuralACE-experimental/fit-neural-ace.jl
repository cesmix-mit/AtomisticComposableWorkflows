using AtomsBase
using InteratomicPotentials 
using InteratomicBasisPotentials
using PotentialLearning
using PotentialLearning: NNBasisPotential, potential_energy, force # TODO: move to InteratomicBasisPotentials.jl
using LinearAlgebra
using Flux
using Optimization
using OptimizationOptimJL
using Random

# Load input parameters
args = ["experiment_path",      "nace-HfO2_cpmd_1000/", #"nace-HfB2/",
        "dataset_path",         "../../../data/",
        "dataset_filename",     "HfO2_cpmd_1000.xyz", #"HfB2-n24-585.exyz",
        "random_seed",          "0",   # Random seed to ensure reproducibility of loading and subsampling.
        "split_prop",           "0.8", # 80% training, 20% test.
        "max_train_sys",        "800", # Subsamples up to 800 systems from the training dataset.
        "max_test_sys",         "200", # Subsamples up to 200 systems from the test dataset.
        "nn",                   "Chain(Dense(n_desc,3,Flux.relu),Dense(3,1))",
        "n_epochs",             "1",
        "n_batches",            "1",
        "optimiser",            "BFGS",
        "max_it",               "1000",
        "n_body",               "2",
        "max_deg",              "3",
        "r0",                   "1.0",
        "rcutoff",              "5.0",
        "wL",                   "1.0",
        "csp",                  "1.0",
        "w_e",                  "1.0",
        "w_f",                  "1.0"]
args = length(ARGS) > 0 ? ARGS : args
input = get_input(args)


# Create experiment folder
path = input["experiment_path"]
run(`mkdir -p $path`)
@savecsv path input

# Fix random seed
if "random_seed" in keys(input)
    Random.seed!(input["random_seed"])
end

# Load datasets
train_sys, e_train, f_train_v, s_train,
test_sys, e_test, f_test_v, s_test = load_datasets(input)


# Subsample datasets
max_train_sys = input["max_train_sys"]; max_test_sys = input["max_test_sys"]
train_sys, e_train, f_train_v, s_train =
    random_subsample(train_sys, e_train, f_train_v, s_train, max_sys = max_train_sys)
test_sys, e_test, f_test_v, s_test =
    random_subsample(test_sys, e_test, f_test_v, s_test, max_sys = max_test_sys)


# Linearize forces
f_train, f_test = linearize_forces.([f_train_v, f_test_v])


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
atomic_symbols = unique(atomic_symbol(first(train_sys)))
ibp_params = ACEParams(atomic_symbols, n_body, max_deg, wL, csp, r0, rcutoff)
@savevar path ibp_params


# Calculate descriptors. TODO: add this to InteratomicBasisPotentials.jl?
calc_B(pars, sys)  = evaluate_basis.(sys, [pars])
calc_dB(pars, sys) = [ Vector(dBs_comp) for dBs_sys in evaluate_basis_d.(sys, [pars])'
                                        for dBs_atom in dBs_sys
                                        for dBs_comp in eachrow(dBs_atom)]
B_time = @time @elapsed B_train = calc_B(ibp_params, train_sys)
dB_time = @time @elapsed dB_train = calc_dB(ibp_params, train_sys)
B_test = calc_B(ibp_params, test_sys)
dB_test = calc_dB(ibp_params, test_sys)
B_train_ext = vcat([ fill(B_train[i], 3length(position(s)))
                     for (i,s) in enumerate(train_sys)]...)
B_test_ext = vcat([ fill(B_test[i], 3length(position(s)))
                    for (i,s) in enumerate(test_sys)]...)
@savevar path B_train
@savevar path dB_train
@savevar path B_test
@savevar path dB_test


# Define neural network model
n_desc = length(first(B_test))
nn = eval(Meta.parse(input["nn"])) # e.g. Chain(Dense(n_desc,2,Flux.relu), Dense(2,1))
nn_params = Flux.params(nn)
n_params = sum(length, Flux.params(nn))
nnbp = NNBasisPotential(nn, nn_params, ibp_params)


# Define batches
n_batches = input["n_batches"]
train_loader_e, train_loader_f, test_loader_e, test_loader_f = 
       get_batches(n_batches, B_train, B_train_ext, e_train, dB_train, f_train,
                   B_test, B_test_ext, e_test, dB_test, f_test)


# Train
println("Training energies and forces...")
epochs = input["n_epochs"]
opt = @eval $(Symbol(input["optimiser"]))()
max_it = input["max_it"]
w_e, w_f = input["w_e"], input["w_f"]
time_fitting = 
    @time @elapsed train_losses_epochs, test_losses_epochs, train_losses_batches = 
            train!( train_loader_e, train_loader_f, test_loader_e, test_loader_f,
                    w_e, w_f, nnbp, epochs, opt, max_it)

@savevar path train_losses_batches
@savevar path train_losses_epochs
@savevar path test_losses_epochs
@savevar path nnbp.nn_params


# Calculate predictions
e_train_pred = potential_energy.(B_train, [nnbp])
f_train_pred = force.(B_train_ext, dB_train, [nnbp])
e_test_pred = potential_energy.(B_test, [nnbp])
f_test_pred = force.(B_test_ext, dB_test, [nnbp])


# Post-process output: calculate metrics, create plots, and save results
metrics = get_metrics( e_train_pred, e_train, f_train_pred, f_train,
                       e_test_pred, e_test, f_test_pred, f_test,
                       B_time, dB_time, time_fitting)
@savecsv path metrics

e_test_plot = plot_energy(e_test_pred, e_test)
@savefig path e_test_plot

f_test_plot = plot_forces(f_test_pred, f_test)
@savefig path f_test_plot

f_test_cos = plot_cos(f_test_pred, f_test)
@savefig path f_test_cos


