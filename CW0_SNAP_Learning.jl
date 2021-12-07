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
See https://ase.tufts.edu/chemistry/lin/images/FortranMD_TeachersGuide.pdf
"""
function gen_test_atomic_conf(D)
    # Domain
    d = 3.47786
    box = [[d, 0.0, 0.0], [0.0, d, 0.0], [0.0, 0.0, d]] * 1.0u"nm"
    # Boundary conditions
    bcs = [DirichletZero(), DirichletZero(), DirichletZero()]
    # No. of atoms per configuration
    N = 30 #864
    # No. of configurations
    M = 90
    # Element
    elem = elements[:Ar]
    # Define atomic configurations
    atomic_confs = []
    for j in 1:M
        atoms = []
        for i in 1:N
            pos = SVector{D}(rand(D)*d*1.0u"nm"...)
            atom = StaticAtom(pos, elem)
            push!(atoms, atom)
        end
        push!(atomic_confs, FlexibleSystem(box, bcs, atoms))
    end
    return atomic_confs
end

# Define parametric types 
D = 3; T = Float64 # TODO: discuss which parametric types are necessary, define a common policy for all packages 

# Generate test atomic configurations: domain and particles (position, velocity, etc)
atomic_confs = gen_test_atomic_conf(D)

# Generate learning data using Lennard Jones and the atomic configurations
ϵ = 1.657e-21u"J"; σ = 0.34u"nm"; cutoff = 2*σ #0.765u"nm"
lj = LennardJones(ϵ, σ, cutoff)
data = gen_test_data(D, atomic_confs, lj)

# Define target potential: SNAP
rcutfac = cutoff.val; twojmax = 9
inter_pot_atomic_confs = inter_pot_conf(atomic_confs) # TODO: remove after full integration with InteratomicPotentials.jl
snap = SNAP(rcutfac, twojmax, inter_pot_atomic_confs[1]) #TODO: improve interface, do not send a conf as argument

# Define learning problem
lp = SmallSNAPLP(snap, inter_pot_atomic_confs, data, trainingsize = 0.8, fit = [:e])

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

