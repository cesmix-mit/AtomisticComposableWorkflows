# This code will be added to PotentialLearning.jl

using AtomsBase
using InteratomicPotentials 
using InteratomicBasisPotentials
using OrderedCollections
using LinearAlgebra 
using UnitfulAtomic
using Unitful 
using Flux
using Flux.Data: DataLoader
using BSON: @save

include("load-data.jl")

# Load input parameters
function get_input(args)
    if size(args, 1) == 0
        #args = [ "80", "20", "3", "3", "5", "1", "1", "3", "0.1", "aHfO2/", "data/",
        #         "TiO2trainingset.xyz", "a-Hfo2-300K-NVT.extxyz"]
        args = [ "80", "20", "3", "3", "1", "5", "1", "1", "1", "1", "1",
                 "TiO2/", "data/", "TiO2trainingset.xyz", "TiO2testset.xyz"]
    end
    input = OrderedDict()
    input["n_train_sys"]         = parse(Int64, args[1])
    input["n_test_sys"]          = parse(Int64, args[2])
    input["n_body"]              = parse(Int64, args[3])
    input["max_deg"]             = parse(Int64, args[4])
    input["r0"]                  = parse(Float64, args[5])
    input["rcutoff"]             = parse(Float64, args[6])
    input["wL"]                  = parse(Float64, args[7])
    input["csp"]                 = parse(Float64, args[8])
    input["w_e"]                 = parse(Float64, args[9])
    input["w_f"]                 = parse(Float64, args[10])
    input["n_batches"]           = parse(Int64, args[11])
    input["experiment_path"]     = args[12]
    input["dataset_path"]        = args[13]
    input["dataset_filename"]    = args[14]
    if length(input) > 14
        input["trainingset_filename"] = args[14]
        input["testset_filename"]     = args[15]
    end       
    return input
end

# Load dataset
function load_dataset(input)
    experiment_path = input["experiment_path"]
    n_train_sys, n_test_sys = input["n_train_sys"], input["n_test_sys"]
    n_sys = n_train_sys + n_test_sys
    if "dataset_filename" in keys(input)
        filename = input["dataset_path"]*input["dataset_filename"]
        systems, energies, forces, stresses =
                                   load_data(filename, max_entries = n_sys)
        # Split into training and testing
        rand_list = randperm(n_sys)
        train_index, test_index = rand_list[1:n_train_sys], rand_list[n_train_sys+1:n_sys]
        train_systems, train_energies, train_forces, train_stress =
                                     systems[train_index], energies[train_index],
                                     forces[train_index], stresses[train_index]
        test_systems, test_energies, test_forces, test_stress =
                                     systems[test_index], energies[test_index],
                                     forces[test_index], stresses[test_index]
    else
        filename = input["dataset_path"]*input["trainingset_filename"]
        train_systems, train_energies, train_forces, train_stresses =
                load_data(filename, max_entries = n_train_sys)
        filename = input["dataset_path"]*input["testset_filename"]
        test_systems, test_energies, test_forces, test_stresses =
                load_data(filename, max_entries = n_test_sys)
    end
    return train_systems, train_energies, train_forces, train_stress,
           test_systems, test_energies, test_forces, test_stress
end


# Linearize energies and forces
function linearize(train_systems, train_energies, train_forces, train_stresses,
                   test_systems, test_energies, test_forces, test_stresses)
    calc_F(forces) = vcat([vcat(vcat(f...)...) for f in forces]...)
    e_train = train_energies
    f_train = calc_F(train_forces)
    e_test = test_energies
    f_test = calc_F(test_forces)
    return e_train, f_train, e_test, f_test
end


# Split data into batches
function get_batches(n_batches, B_train, B_train_ext, e_train, dB_train, f_train,
                     B_test, B_test_ext, e_test, dB_test, f_test)
    
    bs_train_e = floor(Int, length(B_train) / n_batches)
    train_loader_e   = DataLoader((B_train, e_train), batchsize=bs_train_e, shuffle=true)
    bs_train_f = floor(Int, length(dB_train) / n_batches)
    train_loader_f   = DataLoader((B_train_ext, dB_train, f_train),
                                   batchsize=bs_train_f, shuffle=true)
    println("batchsize_e:", bs_train_e, ", batchsize_f:", bs_train_f)

    bs_test_e = floor(Int, length(B_test) / n_batches)
    test_loader_e   = DataLoader((B_test, e_test), batchsize=bs_test_e, shuffle=true)
    bs_test_f = floor(Int, length(dB_test) / n_batches)
    test_loader_f   = DataLoader((B_test_ext, dB_test, f_test),
                                  batchsize=bs_test_f, shuffle=true)
    println("batchsize_e:", bs_test_e, ", batchsize_f:", bs_test_f)
    
    return train_loader_e, train_loader_f, test_loader_e, test_loader_f
end


