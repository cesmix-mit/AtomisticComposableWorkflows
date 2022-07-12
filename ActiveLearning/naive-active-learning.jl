# Simple active learning workflow. Proof of concept.

using AtomsBase
using InteratomicPotentials 
using InteratomicBasisPotentials
using PotentialLearning
using LinearAlgebra 
using StaticArrays
using UnitfulAtomic
using Unitful 
using Atomistic
using NBodySimulator
using Plots


# TODO: add to InteratomicBasisPotentials.jl?
function InteratomicPotentials.energy_and_force(s::AbstractSystem, p::ACE)
    B = evaluate_basis(s, p.basis_params)
    dB = evaluate_basis_d(s, p.basis_params)
    e = austrip.(B' * p.coefficients * 1u"eV")
    f = [SVector(austrip.(d * p.coefficients .* 1u"eV/Å")...) for d in dB]
    return (; e, f)
end


# TODO: use DFTK.jl
function get_dft_data(input)
    return load_dataset(input)
end


# TODO: analyze MD result and determine if retrain is needed
function retrain(md_res)
    return true
end


# Load input parameters
args = ["experiment_path",      "active-learning-a-HfO2/",
        "dataset_path",         "data/",
        "dataset_filename",     "a-Hfo2-300K-NVT.extxyz",
        "n_train_sys",          "80",
        "n_test_sys",           "20",
        "n_body",               "3",
        "max_deg",              "3",
        "r0",                   "1.0",
        "rcutoff",              "5.0",
        "wL",                   "1.0",
        "csp",                  "1.0",
        "w_e",                  "1.0",
        "w_f",                  "1.0", 
        "steps",                "500",
        "ref_temp",             "300.0",
        "delta_t",              "1.0",
        "delta_step",           "100"]
args = length(ARGS) > 0 ? ARGS : args
input = get_input(args)


# Create experiment folder
path = input["experiment_path"]
run(`mkdir -p $path`)


# Run active learning MD simulation
Δt = input["delta_t"]u"fs"
steps = input["steps"]
Δstep = input["delta_step"]
curr_steps = 0
md_res = []
while curr_steps < steps

    if retrain(md_res)
        # Generate DFT data
        train_sys, e_train, f_train_v, s_train,
        test_sys, e_test, f_test_v, s_train = get_dft_data(input)


        # Linearize forces
        f_train, f_test = linearize_forces.([f_train_v, f_test_v])


        # Define ACE params
        n_body = input["n_body"]
        max_deg = input["max_deg"]
        r0 = input["r0"]
        rcutoff = input["rcutoff"]
        wL = input["wL"]
        csp = input["csp"]
        atomic_symbols = unique(atomic_symbol(first(train_sys)))
        ace_params = ACEParams(atomic_symbols, n_body, max_deg, wL, csp, r0, rcutoff)


        # Calculate descriptors. TODO: Add to PotentialLearning.jl?
        calc_B(sys) = vcat((evaluate_basis.(sys, [ace_params])'...))
        calc_dB(sys) =
            vcat([vcat(d...) for d in evaluate_basis_d.(sys, [ace_params])]...)
        B_train = calc_B(train_sys)
        dB_train = calc_dB(train_sys)
        B_test = calc_B(test_sys)
        dB_test = calc_dB(test_sys)


        # Calculate A and b. TODO: Add to PotentialLearning.jl?
        A = [B_train; dB_train]
        b = [e_train; f_train]


        # Calculate coefficients β. TODO: Add to PotentialLearning.jl?
        w_e, w_f = input["w_e"], input["w_f"]
        Q = Diagonal([w_e * ones(length(e_train));
                      w_f * ones(length(f_train))])
        β = (A'*Q*A) \ (A'*Q*b)
        
        
        # Define ACE
        ace = ACE(β, ace_params)
        
        
        # Calculate predictions
        e_test_pred = B_test * β
        f_test_pred = dB_test * β
        
        
        # Calculate metrics
        e_mae, e_rmse, e_rsq = calc_metrics(e_test_pred, e_test)
        f_mae, f_rmse, f_rsq  = calc_metrics(f_test_pred, f_test)
        
        
        # Analyze metrics, report, and take corrective actions.
        if e_mae > 1.0
            println("Warning: fitting error too high.")
        end
    end


    # Update thermostat
    ref_temp = input["ref_temp"]u"K"
    ν = 10 / Δt # stochastic collision frequency
    thermostat = NBodySimulator.AndersenThermostat(austrip(ref_temp), austrip(ν))


    # Run MD simulation
    if curr_steps == 0
        global curr_steps += Δstep
        init_sys = first(test_sys)
        sim = NBSimulator(Δt, curr_steps, thermostat = thermostat)
        global md_res = simulate(init_sys, sim, ace)
    else
        global curr_steps += Δstep
        sim = NBSimulator(Δt, curr_steps, t₀=get_time(md_res))
        global md_res = simulate(get_system(md_res), sim, ace)
    end

end


# Post-process and save results
savefig(Atomistic.plot_temperature(md_res, 10), path*"temp.svg")
savefig(Atomistic.plot_energy(md_res, 10), path*"energy.svg")
savefig(Atomistic.plot_rdf(md_res, 1.0, Int(0.95 * steps)), path*"rdf.svg")
Atomistic.animate(md_res, path*"anim.gif")


