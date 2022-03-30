using AtomsBase
using GLMakie
using InteratomicPotentials 
using InteratomicBasisPotentials
using LinearAlgebra 
using Random
using StaticArrays
using Statistics 
using UnitfulAtomic
using Unitful 

# Data Loading routine
include("load_data.jl")

lj_energies = []
lj_first_escape = []
ace_energies = []
ace_lj_energies = []
lj = InteratomicPotentials.LennardJones(0.01034, 1.0, 4.0, [:Ar])
ace_first_escape = []
for seed = 1:100
    println("Seed: $seed")
    println("LJ \n")
    file = "LJ_MD/$seed/data.xyz"
    systems, energies, forces, stresses = load_data(file; max_entries = Int(1E6));
    append!(lj_energies, [InteratomicPotentials.potential_energy(s, lj) for s in systems])

##########################################################################################################
    file = "ACE_MD/coeff_1/$seed/data.xyz"
    systems, energies, forces, stresses = load_data(file; max_entries = Int(1E6));
    append!(ace_energies, [InteratomicPotentials.potential_energy(s, lj) for s in systems])

end

using JLD
fe_lj = load("first_escape_1000_lj_temp_065.jld")["first_escape"]
fe_ace = load("first_escape_1000_pace_temp_065.jld")["first_escape"]

using KernelDensity

fe_ace_kde = kde([i for i in fe_ace], boundary = (0.0, 7000))
fe_lj_kde = kde([i for i in fe_lj], boundary = (0.0, 7000))

ace_e_kde = kde([i for i in ace_energies])
lj_e_kde = kde([i for i in lj_energies])


fig = plt.figure(figsize = (12, 8))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(ace_e_kde.x, ace_e_kde.density, label = "ACE")
ax1.plot(lj_e_kde.x, lj_e_kde.density, label = "LJ")
ax1.set_xlabel("Energ of Configuration (eV)")
ax1.set_ylabel("Probability")
ax1.set_title("Distribution of Energies (in terms of LJ) during MD")
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(fe_ace_kde.x, fe_ace_kde.density, label = "ACE")
ax2.plot(fe_lj_kde.x, fe_lj_kde.density, label = "LJ")
ax2.set_xlabel("Time to first rearrangement (ps)")
ax2.set_ylabel("Probability")
ax2.set_title("Disttribution of First Rearrangment Times during MD")
plt.tight_layout()

using GLMakie

size_in_inches = (10, 8)
size_in_pt = size_in_inches .* 72
f = CairoMakie.Figure(resolution = size_in_pt, fontsize = 12)


axes1 = CairoMakie.Axis(f[1, 1], title="MD Configuration Energies (in terms of LJ)", xlabel="Energy (eV)", ylabel="Count")
axes2 = CairoMakie.Axis(f[1, 2], title="First Rearrangment Times during MD", xlabel="time (ps)", ylabel="Count")


CairoMakie.hist!(axes1, [i for i in ace_energies], color = (:red, 0.75))
CairoMakie.hist!(axes1, [i for i in lj_energies], color = (:blue, 0.4))
CairoMakie.hist!(axes2, [i for i in fe_ace], color = (:red, 0.75), label = "ACE")
CairoMakie.hist!(axes2, [i for i in fe_lj], color = (:blue, 0.4), label = "LJ")
CairoMakie.axislegend(axes2)
CairoMakie.resize_to_layout!(f)

CairoMakie.save("energies_first_rearrangment_ace_lj.pdf", f)

