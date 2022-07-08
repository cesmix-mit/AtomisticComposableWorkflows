# Simple active learning workflow. Proof of concept.

using AtomsBase
using InteratomicPotentials 
using InteratomicBasisPotentials
using LinearAlgebra 
using Random
using StaticArrays
using Statistics 
using StatsBase
using UnitfulAtomic
using Unitful 
using Atomistic
using NBodySimulator
using BenchmarkTools
using CSV
using Plots

include("input.jl")


# Load input parameters
input = get_input(ARGS)


# Create experiment folder
path = "ace-"*input["experiment_path"]
run(`mkdir -p $path`)


# TODO: use DFTK.jl
function get_dft_data()
    return load_dataset(input)
end


# TODO: this function should be added to InteratomicBasisPotentials.jl?
function InteratomicPotentials.energy_and_force(s::AbstractSystem, p::ACE)
    B = evaluate_basis(s, ace_params)
    dB = evaluate_basis_d(s, ace_params)
    e = austrip.(B' * p.coefficients * 1u"eV")
    f = [SVector(austrip.(d * p.coefficients .* 1u"eV/Å")...) for d in dB]
    return (; e, f)
end


# TODO: analyze MD result and determine if retrain is needed
function retrain(md_res)
    return true
end


steps = 500
curr_steps = 0
md_res = []
while curr_steps < steps

    if retrain(md_res)
        # Generate DFT data
        train_sys, e_train, f_train_v, s_train,
        test_sys, e_test, f_test_v, s_train = get_dft_data()


        # Linearize forces
        f_train, f_test = linearize_forces.([f_train_v, f_test_v])


        # Define ACE potential
        n_body = 5; max_deg = 4; r0 = 1; rcutoff = 5; wL = 1; csp = 1
        atomic_symbols = unique(atomic_symbol(first(train_sys)))
        ace_params = ACEParams(atomic_symbols, n_body, max_deg, wL, csp, r0, rcutoff)
        ace = ACE(β, ace_params)


        # Calculate descriptors
        calc_B(sys) = vcat((evaluate_basis.(sys, [ace_params])'...))
        calc_dB(sys) =
            vcat([vcat(d...) for d in evaluate_basis_d.(sys, [ace_params])]...)
        B_train = calc_B(train_sys)
        dB_train = calc_dB(train_sys)
        B_test = calc_B(test_systems)
        dB_test = calc_dB(test_systems)


        # Calculate A and b
        A = [B_train; dB_train]
        b = [e_train; f_train]


        # Calculate coefficients β
        w_e, w_f = input["w_e"], input["w_f"]
        Q = Diagonal([w_e * ones(length(e_train));
                      w_f * ones(length(f_train))])
        ace.β = (A'*Q*A) \ (A'*Q*b)
    end


    # Update thermostat
    reference_temp = 300.0u"K"
    ν = 10 / Δt # stochastic collision frequency
    thermostat = NBodySimulator.AndersenThermostat(austrip(reference_temp),
                                                   austrip(ν))


    # Run MD simulation
    Δt = 1.0u"fs"
    if curr_steps == 0
        init_sys = first(test_sys)
        sim = NBSimulator(Δt, curr_steps, thermostat = thermostat)
        md_res = simulate(init_sys, sim, ace)
    else
        sim = NBSimulator(Δt, curr_steps, t₀=get_time(md_res))
        md_res = simulate(get_system(md_res), sim, ace)
    end

    curr_steps += 100

end

# Results
savefig(plot_temperature(result, 10), path*"temp.svg")
savefig(plot_energy(result, 10), path*"energy.svg")
savefig(plot_rdf(result, 1.0, Int(0.95 * steps)), path*"rdf.svg")
animate(result, path*"anim.gif")


