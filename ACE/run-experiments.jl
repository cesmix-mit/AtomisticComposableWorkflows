using IterTools

# Experiment folder
experiments = "experiments/"

# Fitting program
juliafile = "fit-neural-ace.jl"

# Parameter definitions ########################################################

# dataset path
dataset_path = ["../../data/"]

# dataset file
dataset_filename = ["a-Hfo2-300K-NVT.extxyz", "HfO2_cpmd_train_0_94_11.xyz"]

# n_systems: number of atomic configurations
n_systems = 2000:2000

# n_body: body order. N: correlation order (N = n_body - 1)
n_body = 2:5

# max_deg: maximum polynomial degree
max_deg = 3:6 

# r0: An estimate on the nearest-neighbour distance for scaling, JuLIP.rnn() 
#     function returns element specific earest-neighbour distance
r0 = 1:1 # ( rnn(:Hf) + rnn(:O) ) / 2.0 ?

# rin: inner cutoff radius
# rin = 0.65*r0 is the default

# rcutoff or rcut: outer cutoff radius
rcutoff = 4:7

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

# e_weight: energy weight, used during fitting in normal equations
e_weight = [1e-8, 1, 100]

# f_weight: force weight, used during fitting in normal equations
f_weight = [1e-8, 1, 100]


# Run experiments ##############################################################

run(`mkdir $experiments`)
for params in product(dataset_path, dataset_filename, n_systems, n_body,
                      max_deg, r0, rcutoff, wL, csp, e_weight, f_weight)
    print("Launching experiment: $params\n")
    currexp = reduce(*,map(s->"$s"*"-", params[2:end]))[1:end-1]
    run(`mkdir $experiments/$currexp`)
    @async run(Cmd(`nohup julia ../../$juliafile ./ $params`, dir="$experiments/$currexp"));
end

print("Run ./gather-results.sh after all the experiments are finished.\n")
