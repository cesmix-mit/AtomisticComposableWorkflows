# Learning example #############################################################

import Pkg
Pkg.add("Unitful")
Pkg.add("PeriodicTable")
Pkg.add("StaticArrays")
Pkg.add("LinearAlgebra")
Pkg.add("AtomsBase")
Pkg.add(url="git@github.com:cesmix-mit/ElectronicStructure.jl.git")
Pkg.add(url="https://github.com/cesmix-mit/InteratomicPotentials.jl.git", rev="integrated-branch")
Pkg.add(url="https://github.com/cesmix-mit/PotentialLearning.jl.git", rev="refactor")

using Unitful, PeriodicTable, StaticArrays, LinearAlgebra
using AtomsBase
using ElectronicStructure
using InteratomicPotentials
using PotentialLearning


"""
    gen_test_atomic_conf(D)

Generate test atomic configurations
"""
function gen_test_atomic_conf(D, L)
    # Domain
    box = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]] * L
    # Boundary conditions
    bcs = [Periodic(), Periodic(), DirichletZero()]
    # No. of atoms per configuration
    N = 30
    # No. of configurations
    M = 20 
    c = elements[:C]
    # Define atomic configurations
    atomic_confs = []
    for j in 1:M
        atoms = []
        for i in 1:N
            pos = SVector{D}(rand(D)*L...)
            atom = StaticAtom(pos,c)
            push!(atoms, atom)
        end
        push!(atomic_confs, FlexibleSystem(box, bcs, atoms))
    end
    return atomic_confs
end

# Define parametric types
D = 3; L = 1.0u"cm"; E = 1.0u"J"

# Generate atomic configurations: domain and particles (position, velocity, etc)
atomic_confs = gen_test_atomic_conf(D, L)

# Generate learning data
lj = LennardJones(1.657e-21u"J", 0.34u"nm")
data = gen_test_data(D, atomic_confs, lj)

# Define potential
rcutfac = 0.1; twojmax = 2
# TODO: line below is not necessary after full integration with InteratomicPotentials.jl
inter_pot_atomic_confs = inter_pot_conf(atomic_confs)
snap = SNAP(rcutfac, twojmax, inter_pot_atomic_confs[1])

# Define learning problem
lp = SmallSNAPLP(snap, inter_pot_atomic_confs, data)

# Learn :-)
learn(lp, LeastSquaresOpt{D, Float64}())



