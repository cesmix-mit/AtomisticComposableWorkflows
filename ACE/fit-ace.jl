using AtomsBase
using InteratomicPotentials 
using InteratomicBasisPotentials
using PotentialLearning
using LinearAlgebra


# Load input parameters
input = get_input(ARGS)


# Create experiment folder
path = "ace-"*input["experiment_path"]
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


# Define ACE parameters
n_body = input["n_body"]
max_deg = input["max_deg"]
r0 = input["r0"]
rcutoff = input["rcutoff"]
wL = input["wL"]
csp = input["csp"]
atomic_symbols = unique(atomic_symbol(first(train_sys)))
ace_params = ACEParams(atomic_symbols, n_body, max_deg, wL, csp, r0, rcutoff)
@savevar path ace_params


# Calculate descriptors. TODO: add this to PotentialLearning.jl?
calc_B(sys) = vcat((evaluate_basis.(sys, [ace_params])'...))
calc_dB(sys) =
    vcat([vcat(d...) for d in evaluate_basis_d.(sys, [ace_params])]...)
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

# Filter outliers. TODO: add this to PotentialLearning.jl?
#fmean = mean(f_train); fstd = std(f_train)
#non_outliers = fmean - 2fstd .< f_train .< fmean + 2fstd 
#f_train = f_train[non_outliers]
#v = BitVector([ ones(length(e_train)); non_outliers])
#A = A[v , :]


# Calculate coefficients β.  TODO: add this to PotentialLearning.jl?
w_e, w_f = input["w_e"], input["w_f"]
Q = Diagonal([w_e * ones(length(e_train));
              w_f * ones(length(f_train))])
β = (A'*Q*A) \ (A'*Q*b)

end

## Check weights. TODO: add this to PotentialLearning.jl?
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

