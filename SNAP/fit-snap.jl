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
        "rcutfac",              "1.2",
        "radii",                "[1.5, 1.5]",
        "rcut0",                "0.989",
        "weight",               "[1.0, 1.0]",
        "chem_flag",            "false",
        "bzero_flag",           "false",
        "bnorm_flag",           "false"]
args = length(ARGS) > 0 ? ARGS : args
input = get_input(args)


# Create experiment folder
path = input["experiment_path"]
run(`mkdir -p $path`)
@savecsv path input


# Load dataset
train_sys, e_train, f_train_v, s_train,
test_sys, e_test, f_test_v, s_train = load_dataset(input)


# Linearize forces
f_train, f_test = linearize_forces.([f_train_v, f_test_v])


@savevar path e_train
@savevar path f_train
@savevar path e_test
@savevar path f_test


# Define SNAP parameters
n_atoms = length(first(train_sys))
twojmax = input["twojmax"]
rcutfac = input["rcutfac"]
radii = input["radii"]
rcut0 = input["rcut0"]
weight = input["weight"]
chem_flag = input["chem_flag"]
bzero_flag = input["bzero_flag"]
bnorm_flag = input["bnorm_flag"]
atomic_symbols = unique(atomic_symbol(first(train_sys)))
snap_params = SNAPParams(n_atoms, twojmax, atomic_symbols, rcutfac, 0.00,
                         rcut0, radii, weight, chem_flag, bzero_flag)
@savevar path snap_params


# Calculate descriptors. TODO: add this to PotentialLearning.jl?
calc_B(sys) = vcat((evaluate_basis.(sys, [snap_params])'...))
calc_dB(sys) =
    vcat([vcat(d...) for d in evaluate_basis_d.(sys, [snap_params])]...)
B_time = @time @elapsed B_train = calc_B(train_sys)
dB_time = @time @elapsed dB_train = calc_dB(train_sys)
B_test = calc_B(test_sys)
dB_test = calc_dB(test_sys)
@savevar path B_train
@savevar path dB_train
@savevar path B_test
@savevar path dB_test


# Calculate A and b.  TODO: add this to PotentialLearning.jl?
time_fitting = Base.@elapsed begin
A = [B_train; dB_train]
b = [e_train; f_train]


# Calculate coefficients β.  TODO: add this to PotentialLearning.jl?
w_e, w_f = input["w_e"], input["w_f"]
Q = Diagonal([w_e * ones(length(e_train));
              w_f * ones(length(f_train))])
β = (A'*Q*A) \ (A'*Q*b)

end


n_params = size(β,1)
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
