using AtomsBase
using InteratomicPotentials 
using InteratomicBasisPotentials
using PotentialLearning
using LinearAlgebra


# Load input parameters
args = ["experiment_path",      "snap-TiO2/",
        "dataset_path",         "../data/",
        "trainingset_filename", "TiO2trainingset.xyz",
        "testset_filename",     "TiO2testset.xyz",
        "n_train_sys",          "80",
        "n_test_sys",           "20",
        "twojmax",              "4",
        "rcutfac",              "6.0",
        "rmin0",                "0.0",
        "rcut0",                "0.989",
        "radii",                "[6.0, 6.0]",
        "weight",               "[1.0, 1.0]",
        "chem_flag",            "false",
        "bzero_flag",           "false",
        "bnorm_flag",           "false",
        "switch_flag",          "false",
        "wselfall_flag",        "false",
        "prebuilt_flag",        "false"]
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
n_train_sys = input["n_train_sys"]; n_test_sys = input["n_test_sys"]
train_sys, e_train, f_train_v, s_train =
    random_subsample(train_sys, e_train, f_train_v, s_train, max_sys = n_train_sys)
test_sys, e_test, f_test_v, s_test =
    random_subsample(test_sys, e_test, f_test_v, s_test, max_sys = n_test_sys)


# Linearize forces
f_train, f_test = linearize_forces.([f_train_v, f_test_v])


@savevar path e_train
@savevar path f_train
@savevar path e_test
@savevar path f_test


# Define SNAP parameters
n_atoms = length(first(train_sys))
twojmax = input["twojmax"]
species = unique(atomic_symbol(first(train_sys)))
rcutfac = input["rcutfac"]
rmin0 = input["rmin0"]
rcut0 = input["rcut0"]
radii = input["radii"]
weight = input["weight"]
chem_flag = input["chem_flag"]
bzero_flag = input["bzero_flag"]
bnorm_flag = input["bnorm_flag"]
switch_flag = input["switch_flag"]
wselfall_flag = input["wselfall_flag"]
prebuilt_flag = input["prebuilt_flag"]
train_pars = [ SNAPParams(length(s), twojmax, species, rcutfac, rmin0, rcut0,
                          radii, weight, chem_flag, bzero_flag, bnorm_flag,
                          switch_flag, wselfall_flag, prebuilt_flag)
               for s in train_sys ]
test_pars = [ SNAPParams(length(s), twojmax, species, rcutfac, rmin0, rcut0,
                         radii, weight, chem_flag, bzero_flag, bnorm_flag,
                         switch_flag, wselfall_flag, prebuilt_flag)
               for s in test_sys ]
@savevar path first(train_pars)


# Calculate descriptors. TODO: add this to PotentialLearning.jl?
# TODO: fix incorrect energy block calculation of matrix A: n. cols of B_train is different from dB_train
calc_B(sys, pars) = vcat(evaluate_basis.(sys, pars)'...)
calc_dB(sys, pars) = vcat([hcat(evaluate_basis_d(s, p)...)'
                           for (s,p) in zip(sys, pars)]...)
B_time = @time @elapsed B_train = calc_B(train_sys, train_pars)
dB_time = @time @elapsed dB_train = calc_dB(train_sys, train_pars)
B_test = calc_B(test_sys, test_pars)
dB_test = calc_dB(test_sys, test_pars)
@savevar path B_train
@savevar path dB_train
@savevar path B_test
@savevar path dB_test


# Calculate coefficients β
w_e, w_f = input["weight"][1], input["weight"][2]
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

