using LAMMPS
using Unitful
using UnitfulAtomic
using AtomsBase
using Random
using StaticArrays
using LinearAlgebra
using Statistics
using JLD
using InteratomicPotentials, InteratomicBasisPotentials

## Load parameters 
n_body = 2
max_deg = 8
r0 = 1.0
rcutoff = 4.0
wL = 1.0
csp = 1.0
rpi_params = RPIParams([:Ar], n_body, max_deg, wL, csp, r0, rcutoff)
β = load("IAP_coefficient_samples/ace_rand_bayesian_posterior_samples_fitted_var.jld")["β"]


# Run MD and compute First rearrangment times
include("../load_data.jl")
include("define_md_LAMMPS_ace.jl")
Tend = Int(1E6)
dT = 50
Temp = 0.65 * 120
save_dir = "ACE_MD/TEMP/"
first_escape = Float64[]
coeff_list = Int[]
pace_energies = Vector{Float64}[]
for seed = 1:10
    println("seed $seed")
    coeff = randperm(length(β))[1]
    file_dir = "ACE_MD/TEMP/"
    save_β_to_file(β[coeff], rpi_params, "TEMP")

    try
        run_md(Tend, file_dir, save_dir; seed = seed, dT = dT, dt = 0.005, Temp = Temp)
        push!(coeff_list, coeff)
        run(`python3 ACE_MD/to_extxyz.py ACE_MD/TEMP/`)

        file = "ACE_MD/TEMP/data.xyz"
        systems, energies, forces, stresses = load_data(file; max_entries = Int(1E6));
        temp_energies = [InteratomicPotentials.potential_energy(s, lj) for s in systems];
        push!(pace_energies, temp_energies)
        ## Get distances from origin
        pos = position.(systems)
        bias = 5.0 * 1u"Å"
        distances = Vector{Float64}[]
        for (i,p) in enumerate(pos) 
            pp = [p_i .- bias for p_i in p]
            push!(distances, ustrip.(norm.(p_i .- mean(pp) for p_i in pp)) )
        end

        ## Compute first escape 
        try 
            min_atom = argmin.(distances)
            first_not_starting = findfirst(min_atom .!= 13) # Atom # 13 is always the starting center atom.
            push!(first_escape, first_not_starting * 50 * 0.005) # Multiply index by dt = 0.005 and by dT = 100 (we only keep every 100 timesteps)
        catch 
            println("No first escape")
        end
    catch 
        println("Error, MD seed # $seed")#, coeff $coeff")
    end

end

save("saved_data/first_escape_1000_pace_dpp_bayesian_sampling.jld", "first_escape", first_escape)

