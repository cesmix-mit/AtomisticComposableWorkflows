using AtomsBase
using InteratomicPotentials 
using InteratomicBasisPotentials
using PotentialLearning
using LinearAlgebra


# Load input parameters
args = ["experiment_path",      "HfB2-33/",
        "dataset_path",         "../../../data/",
        "dataset_filename",     "HfB2-n24-585.exyz",
        "split_prop",           "0.8", # 80% training, 20% test.
        "max_train_sys",        "800", # Subsamples up to 800 systems from the training dataset.
        "max_test_sys",         "200", # Subsamples up to 200 systems from the test dataset.
        "n_body",               "3",
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
params = ACEParams(atomic_symbols, n_body, max_deg, wL, csp, r0, rcutoff)
@savevar path params


# Calculate descriptors. TODO: add this to PotentialLearning.jl?
calc_B(pars, sys)  = vcat(evaluate_basis.(sys, [pars])'...)
calc_dB(pars, sys) = vcat([hcat(evaluate_basis_d(s, pars)...)' for s in sys]...)
B_time = @time @elapsed B_train = calc_B(params, train_sys)
dB_time = @time @elapsed dB_train = calc_dB(params, train_sys)
B_test = calc_B(params, test_sys)
dB_test = calc_dB(params, test_sys)
@savevar path B_train
@savevar path dB_train
@savevar path B_test
@savevar path dB_test


# Filter outliers. TODO: add this to PotentialLearning.jl?
#fmean = mean(f_train); fstd = std(f_train)
#non_outliers = fmean - 2fstd .< f_train .< fmean + 2fstd 
#f_train = f_train[non_outliers]
#v = BitVector([ ones(length(e_train)); non_outliers])
#A = A[v , :]


# Calculate coefficients β
w_e, w_f = input["w_e"], input["w_f"]
time_fitting = Base.@elapsed β = learn(B_train, dB_train, e_train, f_train, w_e, w_f)
@savevar path β


# Calculate predictions
e_train_pred = B_train * β
f_train_pred = dB_train * β
e_test_pred = B_test * β
f_test_pred = dB_test * β


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

