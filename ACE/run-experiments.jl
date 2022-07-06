using IterTools

# Parameter labels
labels = [  "experiment_path",
            "dataset_path",
            "trainingset_filename",
            "testset_filename",
            "n_train_sys",
            "n_test_sys",
            "n_batches",
            "n_body",
            "max_deg",
            "r0",
            "rcutoff",
            "wL",
            "csp",
            "w_e",
            "w_f"]

# Experiment folder
experiments_path = "experiments/"

# Fitting program
juliafile = "fit-neural-ace.jl"

# Parameter definitions ########################################################

# dataset path
dataset_path = ["data/"]

# datasets filename
trainingset_filename = ["TiO2trainingset.xyz"]
testset_filename = ["TiO2testset.xyz"]

# number of atomic configurations
#n_systems = 100:100
n_train_sys = 80:80
n_test_sys = 20:20

# number of batches per dataset
n_batches = 8:8

# n_body: body order. N: correlation order (N = n_body - 1)
n_body = 2:5

# max_deg: maximum polynomial degree
max_deg = 3:6

# r0: An estimate on the nearest-neighbour distance for scaling, JuLIP.rnn() 
#     function returns element specific earest-neighbour distance
r0 = 1.0:1.0 # ( rnn(:Hf) + rnn(:O) ) / 2.0 ?

# rin: inner cutoff radius
# rin = 0.65*r0 is the default

# rcutoff or rcut: outer cutoff radius
rcutoff = 4.0:7.0

# D: specifies the notion of polynomial degree for which there is no canonical
#    definition in the multivariate setting. Here we use SparsePSHDegree which
#    specifies a general class of sparse basis sets; see its documentation for
#    more details. Default: D = ACE1.SparsePSHDegree(; wL = rpi.wL, csp = rpi.csp)
# wL: ?
wL = 0.5:0.5:1.5
# csp: ?
csp = 0.5:0.5:1.5

# pin: specifies the behaviour of the basis as the inner cutoff radius.
# pin = 0 is the default.

# w_e: energy weight, used during fitting in normal equations
w_e = [1e-8, 1.0, 100.0]

# w_f: force weight, used during fitting in normal equations
w_f = [1e-8, 1.0, 100.0]


# Run experiments ##############################################################

run(`mkdir $experiments_path`)
for params in product(dataset_path, trainingset_filename, testset_filename,
                      n_train_sys, n_test_sys, n_batches,
                      n_body, max_deg, r0, rcutoff, wL, csp, w_e, w_f)
    print("Launching experiment: $params\n")
    currexp_path = reduce(*,map(s->"$s"*"-", params[2:end]))[1:end-1]
    params = "$(labels[1]) $experiments_path/$currexp_path " * 
              reduce(*, ["$l $p " for (l, p) in zip(labels[2:end], params)])

    # Serial execution
    run(Cmd(`julia $juliafile $params`, dir="./"));

    # Parallel execution: if the number of parallel experiments is high it may degrade performance.
    #@async run(Cmd(`nohup julia ../../$juliafile $params`, dir="$experiments_path/$currexp_path"));
end

print("Run ./gather-results.sh after all the experiments are finished.\n")
