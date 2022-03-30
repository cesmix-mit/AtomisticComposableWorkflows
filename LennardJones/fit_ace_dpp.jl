using AtomsBase
using CairoMakie
using DPP
using InteratomicPotentials 
using InteratomicBasisPotentials
using LinearAlgebra 
using Random
using StaticArrays
using Statistics 
using StatsBase
using UnitfulAtomic
using Unitful 

include("load_data.jl")

systems, energies, forces, stresses = load_data("LJ_MD/1/data.xyz"; max_entries = 20000);

# Split into training, testing
N_train = 1000
N_batch = 40
N_test  = 100
rand_list = randperm(length(systems))
train_index, test_index = rand_list[1:N_train], rand_list[N_train+1:N_train+1+N_test]
train_systems, train_energies, train_forces, train_stress = systems[train_index], energies[train_index], forces[train_index], stresses[train_index];
test_systems, test_energies, test_forces, test_stress = systems[test_index], energies[test_index], forces[test_index], stresses[test_index];

## Create RPI Basis (2body, 8 polynomial degree)
n_body = 2
max_deg = 8
r0 = 1.0
rcutoff = 4.0
wL = 1.0
csp = 1.0
rpi_params = RPIParams([:Ar], n_body, max_deg, wL, csp, r0, rcutoff)

## Calculate descriptors 
e = zeros(N_train)
f = zeros(N_train*3*13)

B = zeros(N_train, max_deg)
dB = zeros(N_train*3*13, max_deg)
for i = 1:N_train
    sys = train_systems[i]

    BB = reshape(evaluate_basis(sys, rpi_params), 1, :)
    B[i, :] = BB

    dBB = vcat(evaluate_basis_d(sys, rpi_params)...)
    dB[(i-1)*3*13+1:i*3*13, :] = dBB
    
    e[i] = train_energies[i]
    f[(i-1)*3*13+1:i*3*13] = [fi for fi in vcat(vcat(train_forces[i]...)...)]
end

# Calculate Kernel 
mean_B = mean(Bvec)
cov_B = cov(Bvec)

# Feature Kernel
kernel(x, y) = exp(-0.5*(x-y)'*(cov_B\(x-y))) # - 0.5 * logdet(cov_B) - 0.5 * max_deg * log(2*pi))
K = hcat( [[kernel(x, y) for x in Bvec] for y in Bvec]... )

# Matrix Kernel
A = [B; dB]
C = inv(A'*A)
kernel1(x, y) = exp(-0.5*(x-y)'*(C*(x-y)) * 10000.0)
K1 = hcat( [[kernel1(x, y) for x in Bvec] for y in Bvec]... )

# Calculate DPP 
L = EllEnsemble(K)
rescale!(L, 20) # Set batch size to 20
indices = DPP.sample(L, 20)

L1 = EllEnsemble(K1)
indices = DPP.sample(L1, 20)

# Estimate β
function cost(σe, σf)
    indices = DPP.sample(L1, 20)
    A = [B[indices, :]; dB[indices, :]]
    b = [e[indices]; f[indices]]
    Q = Diagonal([ exp(σe) .+ 0.0 * e[indices]; exp(σf) .+ 0.0*f[indices] ])
    K = (A'*Q*A)
    y = A'*Q*b
    β = K \ y 

    A = [B; dB]
    Q = Diagonal([ exp(σe) .+ 0.0 * e; exp(σf) .+ 0.0*f ])

    b = [e; f]

    # -0.5 * (A*β - b)' * Q * (A*β - b) - 0.5 * logdet(Q)
    norm(A*β - b)
end

σe = -1:0.2:5
σf = -1:0.2:5


β_batches = Vector{Float64}[]
for i = 1:100
    indices = DPP.sample(L1, 20)
    A = [B[indices, :]; dB[indices, :]]
    b = [e[indices]; f[indices]]
    Q = Diagonal([0.5 .+ 0.0 * e[indices]; 90.0 .+ 0.0*f[indices] ])
    β_temp = (A'*Q*A) \ (A'*Q*b) 
    # β_temp = A \ b
    push!(β_batches, β_temp)
end

mean_β = mean(β_batches)
cov_β = cov(β_batches)
e_pred = [ B * β for β in β_batches]
f_pred = [ dB * β for β in β_batches ]


rpi_batches = RPI.(β_batches, [rpi_params for i in β_batches])

mean_e_error = Float64[]
var_e_error = Float64[]
mean_f_error = Float64[]
var_f_error = Float64[]
for j = 1:N_test
    B_test_batches = evaluate_basis.([test_systems[j] for i in β_batches], [rpi_params for i in β_batches])
    dB_test_batches = evaluate_basis_d.([test_systems[j] for i in β_batches], [rpi_params for i in β_batches])

    e_pred = @. dot(B_test_batches, β_batches)
    e_true = test_energies[j]
    e_error = abs.(e_pred .- e_true) / abs(e_true)
    push!(mean_e_error, mean(e_error))
    push!(var_e_error, var(e_error))

    f_error = [ mean(abs.(vcat(dB_test_batches[k]...) * β_batches[k] .- vcat(test_forces[j]...)) / abs.(vcat(test_forces[j]...))) for k = 1:length(β_batches)]
    push!(mean_f_error, mean(f_error))
    push!(var_f_error, var(f_error))
end



## Export ACE 
# Remember to change parameters.ace ACE1.julia -> ACE
using ACE1
using JuLIP
for (i, β) in enumerate(β_batches)
    try
        mkdir("ACE_MD/coeff_$i/")
    catch
        
    end
    basis = get_rpi(rpi_params)
    IP = JuLIP.MLIPs.combine(basis, β)
    ACE1.Export.export_ace("ACE_MD/coeff_$i/parameters.ace", IP)
end

include("ACE_MD/run_ace_md.jl")