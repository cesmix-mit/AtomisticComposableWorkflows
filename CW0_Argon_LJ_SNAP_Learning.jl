# SNAP learning/fitting example
#
# It shows a first integration of the following packages under development:
# AtomsBase.jl, ElectronicStructure.jl, InteratomicPotentials.jl, and PotentialLearning.jl
#

#import Pkg
#Pkg.add("Unitful")
#Pkg.add("PeriodicTable")
#Pkg.add("StaticArrays")
#Pkg.add("LinearAlgebra")
#Pkg.add("AtomsBase")
#Pkg.add(url="git@github.com:cesmix-mit/ElectronicStructure.jl.git")
#Pkg.add(url="https://github.com/cesmix-mit/InteratomicPotentials.jl.git", rev="integrated-branch")
#Pkg.add(url="https://github.com/cesmix-mit/PotentialLearning.jl.git", rev="refactor")

using Unitful, PeriodicTable, StaticArrays, LinearAlgebra
using AtomsBase
using ElectronicStructure
using InteratomicPotentials
using PotentialLearning

"""
    gen_test_atomic_conf(D)

Generate test atomic configurations.
"""
function gen_test_atomic_conf(D, σ)
    # Domain
    σ0 = 2.0 * σ; d = 5.0*σ
    box = [[d, 0.0u"nm", 0.0u"nm"], [0.0u"nm", d, 0.0u"nm"], [0.0u"nm", 0.0u"nm", d]]
    # Boundary conditions
    bcs = [DirichletZero(), DirichletZero(), DirichletZero()]
    # No. of atoms per configuration
    N = 2
    # No. of configurations
    M = 40
    # Element
    elem = elements[:Ar]
    # Define atomic configurations
    atomic_confs = []
    for j in 1:M
        atoms = []
        ϕ = rand() * 2.0 * π; θ = rand() * π
        x = (d/2.0-σ0) * cos(ϕ) * sin(θ) + d/2.0
        y = (d/2.0-σ0) * sin(ϕ) * sin(θ) + d/2.0
        z = (d/2.0-σ0) * cos(θ) + d/2.0
        pos1 = SVector{D}(x, y, z)
        atom = StaticAtom(pos1, elem)
        push!(atoms, atom)
        ϕ = rand() * 2.0 * π; θ = rand() * π
        x += σ0 * cos(ϕ) * sin(θ)
        y += σ0 * sin(ϕ) * sin(θ)
        z += σ0 * cos(θ)
        pos2 = SVector{D}(x, y, z)
        atom = StaticAtom(pos2, elem)
        push!(atoms, atom)
        push!(atomic_confs, FlexibleSystem(box, bcs, atoms))
    end
    return atomic_confs
end

# Define parameters
D = 3; T = Float64 # TODO: discuss which parametric types are necessary, define a common policy for all packages 
σ = 1.0u"nm"

# Generate test atomic configurations: domain and particles (position, velocity, etc)
atomic_confs = gen_test_atomic_conf(D, σ)

# Generate learning data using Lennard Jones and the atomic configurations
ϵ = 1.0u"J"; cutoff = 2.5*σ
lj = LennardJones(ϵ, σ, cutoff)
data = gen_test_data(D, atomic_confs, lj)

# Define target potential: SNAP
rcutfac = cutoff.val; twojmax = 6
inter_pot_atomic_confs = inter_pot_conf(atomic_confs) # TODO: remove after full integration with InteratomicPotentials.jl
snap = SNAP(rcutfac, twojmax, inter_pot_atomic_confs[1]) #TODO: improve interface, do not send a conf as argument

# Define learning problem
lp = SmallSNAPLP(snap, inter_pot_atomic_confs, data, trainingsize = 0.8, fit = [:e, :f])

# Learn :-)
learn(lp, LeastSquaresOpt{D, T}())
#learn(lp, QRLinearOpt{D, T}())
#learn(lp, β_loss, NelderMeadOpt{D, T}(1000))

# Validation
y_approx = lp.A_val * lp.snap.β
max_rel_error = maximum(abs.(lp.y_val - y_approx) ./ lp.y_val)
max_abs_error = maximum(abs.(lp.y_val - y_approx))
println("Maximum relative error: ", max_rel_error)
println("Maximum absolute error: ", max_abs_error)

