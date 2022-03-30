using LAMMPS
using Unitful
using UnitfulAtomic
using AtomsBase
using StaticArrays
using LinearAlgebra
using Statistics
using JLD
using InteratomicPotentials

include("../load_data.jl")

Tend = Int(1.5E6)
dT = 50
Temp = 0.65 * 120
save_dir = "LJ_MD/TEMP/"
first_escape = Float64[]
for seed = 1:1000
    println("seed $seed")
    try 
        run_md(Tend, save_dir; seed = seed, dT = dT, dt = 0.005, Temp = Temp)
        run(`python3 LJ_MD/to_extxyz.py LJ_MD/TEMP/`)


    
        file = "LJ_MD/TEMP/data.xyz"
        systems, energies, forces, stresses = load_data(file; max_entries = Int(1E6));

        ## Get distances from origin
        pos = position.(systems)
        bias = 5.0 * 1u"Å"
        distances = Vector{Float64}[]
        for (i,p) in enumerate(pos) 
            pp = [p_i .- bias for p_i in p]
            push!(distances, ustrip.(norm.(p_i .- mean(pp) for p_i in pp)) )
        end
        # plt.plot( hcat(distances...)')

        ## Compute first escape 
        try 
            min_atom = argmin.(distances)
            first_not_starting = findfirst(min_atom .!= 13) # Atom # 13 is always the starting center atom.
            push!(first_escape, first_not_starting * 50 * 0.005) # Multiply index by dt = 0.005 and by dT = 100 (we only keep every 100 timesteps)
        catch 
            println("No first escape")
        end
    catch
        println("MD error, seed # $(seed)")
    end
end


