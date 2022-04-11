using LinearAlgebra
using StaticArrays
using AtomsBase
using InteratomicPotentials 
using InteratomicBasisPotentials
using Atomistic
using NBodySimulator
using BenchmarkTools
using Plots

include("load_data.jl")

experiment_path = "md-ahfo2-ace-nbs/"
run(`mkdir -p $experiment_path`)

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
β = [-1072.1312864239615, -415.4169141829893, -71.953485409436, 1958.9574772187705,
     -281.6558117872044, 630.2363590775907, -2041.0185474208306, 240.9014951656523,
     -660.3623178315258, 267.9793847831617, 129.99873251077383, 43.761861053870035]
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

savefig(plot_temperature!(temp, prod_result, 10), experiment_path*"temp_hfo2_ace_nbs.svg")
savefig(plot_energy!(energy, prod_result, 10), experiment_path*"energy_hfo2_ace_nbs.svg")

rdf = plot_rdf(prod_result, 1.0, Int(0.95 * prod_steps))
savefig(rdf, experiment_path*"rdf_hfo2_ace_nbs_rdf.svg")

animate(prod_result, experiment_path*"hfo2_ace_nbs.gif")

