using LinearAlgebra
using StaticArrays
using AtomsBase
using InteratomicPotentials 
using InteratomicBasisPotentials
using Atomistic
using NBodySimulator
using BenchmarkTools
#using ThreadsX #julia --threads 4
using Plots

include("load_data.jl")


# System #######################################################################
systems, energies, forces, stresses = load_data("data/a-Hfo2-300K-NVT.extxyz")
N = length(systems[1])
initial_system = systems[1]
Δt = 2.5 #2.5u"ps"


# Thermostat ###################################################################
reference_temp = 300.0#u"K"
ν = 0.0001 / Δt # stochastic collision frequency
#eq_thermostat = NBodySimulator.AndersenThermostat(austrip(reference_temp), austrip(ν))
eq_thermostat = NBodySimulator.AndersenThermostat(reference_temp, ν)


# Create potential #############################################################
n_body = 2
max_deg = 3
r0 = 1
rcutoff = 5
wL = 1
csp = 1
β = [-1070.3349953499962, -414.80248420688645, -71.86787965958288, -82.50976688678328,
     -13.689747506835308, -40.72945576866673, -1.3835599761720792, -27.793686953761107,
      10.48034807957747, 275.7704248933768, 132.03456243001267, 43.911546111453454]
rpi_params = RPIParams([:Hf, :O], n_body, max_deg, wL, csp, r0, rcutoff)
potential = RPI(β, rpi_params)

function InteratomicPotentials.energy_and_force(s::AbstractSystem, p::RPI)
    B = evaluate_basis(s, rpi_params)
    dB = evaluate_basis_d(s, rpi_params)
    e = B' * p.coefficients
    f = [SVector(d * p.coefficients...) for d in dB]
    return (; e, f)
end


# First stage ##################################################################
eq_steps = 100
eq_simulator = NBSimulator(Δt, eq_steps, thermostat = eq_thermostat)
eq_result = @time simulate(initial_system, eq_simulator, potential)


# Second stage #################################################################
prod_steps = 100
prod_simulator = NBSimulator(Δt, prod_steps, t₀ = austrip(get_time(eq_result)))
prod_result = @time simulate(get_system(eq_result), prod_simulator, potential)


# Results ######################################################################
temp = plot_temperature(eq_result, 10)
energy = plot_energy(eq_result, 10)

savefig(plot_temperature!(temp, prod_result, 10), "temp_hfo2_ace_nbs.svg")
savefig(plot_energy!(energy, prod_result, 10), "energy_hfo2_ace_nbs.svg")

rdf = plot_rdf(prod_result, 1.0, Int(0.95 * prod_steps))
savefig(rdf, "rdf_hfo2_ace_nbs_rdf.svg")

animate(prod_result, "hfo2_ace_nbs.gif")

