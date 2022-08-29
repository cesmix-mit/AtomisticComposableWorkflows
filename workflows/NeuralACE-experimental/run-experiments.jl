# Run multiple fitting experiments in serial or parallel.
#
# 1. Update parameters ranges in run-experiments.jl
# 2. Run: $ julia --project=../../ run-experiments.jl
# 3. After all experiments have been completed, run the following script to gather
#    the results into a single csv: $ ./gather-results.sh
#

using IterTools

# Parameter labels
labels = [  "experiment_path",
            "dataset_path",
            "dataset_filename",
            "random_seed",
            "split_prop",
            "max_train_sys",
            "max_test_sys",
            "nn",
            "n_epochs",
            "n_batches",
            "optimiser",
            "max_it",
            "n_body",
            "max_deg",
            "r0",
            "rcutoff",
            "wL",
            "csp",
            "w_e",
            "w_f"]

# Parallel execution. Warning: a high number of parallel experiments may degrade system performance.
parallel = true

# Experiment folder
experiments_path = "experiments/"

# Fitting program
juliafile = "fit-neural-ace.jl"

# Parameter definitions ########################################################

# dataset path
dataset_path = ["../../../data/"]

# dataset filename
dataset_filename = [ "HfB2-n24-585.exyz",
                     "HfO2_relax_1000_989.xyz",
                     "HfO2_cpmd_1000.xyz",
                     "HfO2_cpmd_train_0_94_11_7700.xyz",
                     "GaN_md_150K_32atom_2000.extxyz"]

# random_seed: random seed to ensure reproducibility of loading and subsampling.
#              The length of this vector determines the no. of repetitions of each experiment.
random_seed = [123, 345]

# split_prop: split proportion. E.g. 0.8 for training, 0.2 for test.
split_prop = [0.8]

# max_train_sys, max_test_sys: max. no. of atomic conf. in training and test
max_train_sys = [800]
max_test_sys =  [200]

# nn: neural network model
nn = ["Chain(Dense(n_desc,1,Flux.relu),Dense(1,1))",
      "Chain(Dense(n_desc,2,Flux.relu),Dense(2,1))"]

# n_epochs: no. of epochs
n_epochs = [1]

# n_batches: no. of batches per dataset
n_batches = [1]

# optimiser: optimiser of the neural network model. E.g. ADAM, BFGS.
optimiser = ["BFGS"]

# max_it: max. no. of iterations of the optimizer
max_it = [1200]

# n_body: body order. N: correlation order (N = n_body - 1)
n_body = [3]

# max_deg: maximum polynomial degree
max_deg = [3]

# r0: An estimate on the nearest-neighbour distance for scaling, JuLIP.rnn() 
#     function returns element specific earest-neighbour distance
r0 = [1.0] # ( rnn(:Hf) + rnn(:O) ) / 2.0 ?

# rin: inner cutoff radius
# rin = 0.65*r0 is the default

# rcutoff or rcut: outer cutoff radius
rcutoff = [5.0]

# D: specifies the notion of polynomial degree for which there is no canonical
#    definition in the multivariate setting. Here we use SparsePSHDegree which
#    specifies a general class of sparse basis sets; see its documentation for
#    more details. Default: D = ACE1.SparsePSHDegree(; wL = rpi.wL, csp = rpi.csp)
# wL: ?
wL = [1.0]
# csp: ?
csp = [1.0]

# pin: specifies the behaviour of the basis as the inner cutoff radius.
# pin = 0 is the default.

# w_e: energy weight, used during fitting in normal equations
w_e = [1.0]

# w_f: force weight, used during fitting in normal equations
w_f = [1.0]


# Run experiments ##############################################################

run(`mkdir -p $experiments_path`)
for params in product(dataset_path, dataset_filename, random_seed, split_prop,
                      max_train_sys, max_test_sys, nn, n_epochs, n_batches,
                      optimiser, max_it, n_body, max_deg, r0, rcutoff, wL,
                      csp, w_e, w_f)
    print("Launching experiment with parameters: ")
    currexp_path = reduce(*,map(s->"$s"*"-", params[2:end]))[1:end-1]
    params = vcat(["$(labels[1])", "$experiments_path$currexp_path/"],
                   vcat([ ["$l", "$p"] for (l, p) in zip(labels[2:end], params)]...))
    println("$params")
    
    if parallel
        @async run(Cmd(`nohup julia --project=../../ $juliafile $params`, dir="./"));
    else
        run(Cmd(`julia --project=../../ $juliafile $params`, dir="./"));
    end

    println("")
end

print("Run ./gather-results.sh after all the experiments are finished.\n")
