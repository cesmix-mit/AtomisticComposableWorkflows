### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ 1247fb54-dbe8-4721-958b-5d615ce24032
begin
	import Pkg
	Pkg.Registry.add(
		Pkg.RegistrySpec(url = "https://github.com/cesmix-mit/CESMIX.git"))
end

# ╔═╡ c7c5a782-2d3e-4409-aff0-f859c4242597
begin
	Pkg.add("InteratomicPotentials")
	Pkg.add("AtomsBase")
	Pkg.add("UnitfulAtomic")
	Pkg.add("Unitful")
	Pkg.add("StaticArrays")
	Pkg.add("LinearAlgebra") 
	Pkg.add("Statistics")
end

# ╔═╡ 7d9b8895-0dd4-4b8f-9b15-b45044f8956d
begin
	using InteratomicPotentials 
	using AtomsBase
	using UnitfulAtomic
	using Unitful 
	using StaticArrays
	using LinearAlgebra 
	using Statistics 
end

# ╔═╡ e2fb6a18-3da5-497e-9787-bb4af2647b3f
md"# [WIP] Fitting energy and forces using SNAP"

# ╔═╡ 99bab0a6-5bbb-4c48-9ff6-1444abd3a6dc
md"Notebook based on the [this example](https://github.com/cesmix-mit/InteratomicPotentials.jl/tree/main/examples/LJCluster) from InteratomicPotentials.jl, and [this example](https://cesmix-mit.github.io/LAMMPS.jl/dev/generated/fitting_snap/) from LAMMPS.jl."

# ╔═╡ 73ae3300-fbb3-4a7d-a055-3f995ad55071
md"### Installing packages and loading modules"

# ╔═╡ b144a894-fb60-4918-8368-11c8ab6586e8
md"### Loading curated Lennard-Jones cluster data."

# ╔═╡ 311cc38a-8b9a-11ec-362f-edac4da039d4
function load_data(;num_entries = 2000, file = "curated_lj_cluster.xyz" )
    systems  = Vector{AbstractSystem}(undef, num_entries)
    energies = Vector{Float64}(undef, num_entries)
    forces    = Vector{Vector{T} where T<:SVector{3, <:Float64}}(undef, num_entries)
    bias = @SVector [4.0, 4.0, 4.0]
    box = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 9.0]] * 1u"Å"
    bcs = [DirichletZero(), DirichletZero(), DirichletZero()]
    open(file, "r") do io
        count = 1
        while !eof(io) && (count <= num_entries)
            line = readline(io)
            num_atoms = parse(Int, line)
            info_line = split(readline(io))
            energies[count] = parse(Float64, info_line[2][8:end])

            atoms = Vector{Atom}(undef, 13)
            force = Vector{SVector{3, Float64}}(undef, 13) 
            for i = 1:num_atoms
                line = split(readline(io))
                element = Symbol(line[1])
                position = @SVector [ parse(Float64, line[2]), parse(Float64, line[3]), parse(Float64, line[4]) ]
                position += bias
                atoms[i] = Atom(element, position * 1u"Å") 

                force[i] = @SVector [ parse(Float64, line[5]), parse(Float64, line[6]), parse(Float64, line[7])]
            end

            forces[count] = force
            systems[count] = FlexibleSystem(atoms, box, bcs)
            count += 1
        end
        println(count)
    end
    return systems, energies, forces 
end

# ╔═╡ 6ea8ce57-5944-4e20-a716-0c71f8f0ad42
systems, energies, forces = load_data(;num_entries = 100);

# ╔═╡ ca9582f1-d10d-466d-a809-e7439521ea78
md"### Split data into training, testing."

# ╔═╡ fb794e2f-1489-4574-b014-a58d6afce3c3
begin
	train_systems, train_energies, train_forces = 
								systems[1:50], energies[1:50], forces[1:50];
	test_systems, test_energies, test_forces =
								systems[51:end], energies[51:end], forces[51:end];
end

# ╔═╡ 9d42b0ae-24bc-47c2-9481-c1aec9e819e8
begin
	## Create SNAP Basis
	n_atoms = 13
	twojmax = 5
	species = [:Ar]
	rcutfac = 4.0
	rmin0 = 0.0
	rfac0 = 0.0
	radii = [0.0]
	weight = [0.0]
	snap_params = SNAPParams( n_atoms, twojmax, species, rcutfac,
		                      rmin0, rfac0, radii, weight)
	
	## Calculate descriptors 
	train_descriptors = [evaluate_basis(sys, snap_params) for sys in train_systems]
	test_descriptors = [evaluate_basis(sys, snap_params) for sys in test_systems]
	
end

# ╔═╡ e50d67d9-a00c-4f6f-94f4-d0e4e7b88103
md"### Estimate β"

# ╔═╡ eb3fefe7-c682-406b-82fa-85f62ffa3dbc
β = hcat(train_descriptors...)' \ train_energies

# ╔═╡ ca2db5fc-3939-4b05-aa6c-f319283d9ce6
md"### Create and SNAP instance"

# ╔═╡ 6d56e2a6-20c1-4b7b-bd26-fda7660e7b5e
snap = SNAP(β, snap_params)

# ╔═╡ 43afde32-d282-4b85-8ffb-78b9202e8d2d
md"### Test"

# ╔═╡ 8a8cfeac-2c66-4a13-93ec-9e70db384fae
potential_energy(test_systems[1], snap), test_energies[1]

# ╔═╡ Cell order:
# ╟─e2fb6a18-3da5-497e-9787-bb4af2647b3f
# ╟─99bab0a6-5bbb-4c48-9ff6-1444abd3a6dc
# ╠═73ae3300-fbb3-4a7d-a055-3f995ad55071
# ╠═1247fb54-dbe8-4721-958b-5d615ce24032
# ╠═c7c5a782-2d3e-4409-aff0-f859c4242597
# ╠═7d9b8895-0dd4-4b8f-9b15-b45044f8956d
# ╟─b144a894-fb60-4918-8368-11c8ab6586e8
# ╠═311cc38a-8b9a-11ec-362f-edac4da039d4
# ╠═6ea8ce57-5944-4e20-a716-0c71f8f0ad42
# ╟─ca9582f1-d10d-466d-a809-e7439521ea78
# ╠═fb794e2f-1489-4574-b014-a58d6afce3c3
# ╠═9d42b0ae-24bc-47c2-9481-c1aec9e819e8
# ╠═e50d67d9-a00c-4f6f-94f4-d0e4e7b88103
# ╠═eb3fefe7-c682-406b-82fa-85f62ffa3dbc
# ╟─ca2db5fc-3939-4b05-aa6c-f319283d9ce6
# ╠═6d56e2a6-20c1-4b7b-bd26-fda7660e7b5e
# ╠═43afde32-d282-4b85-8ffb-78b9202e8d2d
# ╠═8a8cfeac-2c66-4a13-93ec-9e70db384fae
