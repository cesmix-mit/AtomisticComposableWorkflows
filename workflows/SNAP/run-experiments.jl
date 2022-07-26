# Run multiple fitting experiments in serial or parallel.
#
# 1. Update parameters ranges in run-experiments.jl
# 2. Run: $ julia run-experiments.jl
# 3. After all experiments have been completed, run the following script to gather
#    the results into a single csv: $ ./gather-results.sh
#

using IterTools

# Parameter labels
labels = [  "experiment_path",
            "dataset_path",
            "trainingset_filename",
            "testset_filename",
            "n_train_sys",
            "n_test_sys",
            "twojmax",
            "rcutfac",
            "radii",
            "rcut0",
            "weight",
            "chem_flag",
            "bzero_flag",
            "bnorm_flag"]


# Parallel execution. Warning: a high number of parallel experiments may degrade system performance.
parallel = false

# Experiment folder
experiments_path = "experiments/"

# Fitting program
juliafile = "fit-snap.jl"

# Parameter definitions ########################################################

# dataset path
dataset_path = ["../data/"]

# datasets filename
trainingset_filename = ["TiO2trainingset.xyz"]
testset_filename = ["TiO2testset.xyz"]

# number of atomic configurations
n_train_sys = 80:80
n_test_sys = 20:20

# TODO: COMPLETE
twojmax = 1:5
rcutfac = 1.0:1.5
radii = "[1.5, 1.5]"
rcut0 = 0.989:0.989
weight = "[1.0, 1.0]"
chem_flag = "false"
bzero_flag = "false"
bnorm_flag = "false"


# Run experiments ##############################################################

run(`mkdir $experiments_path`)
for params in product(dataset_path, trainingset_filename, testset_filename,
                      n_train_sys, n_test_sys, n_batches,
                      n_body, max_deg, r0, rcutoff, wL, csp, w_e, w_f)
    print("Launching experiment with parameters: ")
    currexp_path = reduce(*,map(s->"$s"*"-", params[2:end]))[1:end-1]
    params = vcat(["$(labels[1])", "$experiments_path$currexp_path/"],
                   vcat([ ["$l", "$p"] for (l, p) in zip(labels[2:end], params)]...))
    println("$params")
    
    if parallel
        @async run(Cmd(`nohup julia $juliafile $params`, dir="./"));
    else
        run(Cmd(`julia $juliafile $params`, dir="./"));
    end

    println("")
end

print("Run ./gather-results.sh after all the experiments are finished.\n")
