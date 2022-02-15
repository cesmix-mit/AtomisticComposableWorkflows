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
	Pkg.add("InteratomicBasisPotentials")	
	Pkg.add("AtomsBase")
	Pkg.add("UnitfulAtomic")
	Pkg.add("Unitful")
	Pkg.add("StaticArrays")
	Pkg.add("LinearAlgebra") 
	Pkg.add("Statistics")
	Pkg.add("DataFrames")
	Pkg.add("Plots")
end

# ╔═╡ 7d9b8895-0dd4-4b8f-9b15-b45044f8956d
begin
	using InteratomicPotentials 
	using InteratomicBasisPotentials 
	using AtomsBase
	using UnitfulAtomic
	using Unitful 
	using StaticArrays
	using LinearAlgebra 
	using Statistics 
	using DataFrames
	using Plots
end

# ╔═╡ e2fb6a18-3da5-497e-9787-bb4af2647b3f
md"# [WIP] Fitting energy and forces using SNAP"

# ╔═╡ 99bab0a6-5bbb-4c48-9ff6-1444abd3a6dc
md"Notebook based on the [this example](https://github.com/cesmix-mit/InteratomicPotentials.jl/tree/main/examples/LJCluster) from InteratomicPotentials.jl, and [this example](https://cesmix-mit.github.io/LAMMPS.jl/dev/generated/fitting_snap/) from LAMMPS.jl."

# ╔═╡ 73ae3300-fbb3-4a7d-a055-3f995ad55071
md"### Installing packages and loading modules"

# ╔═╡ a5ef415f-b3a3-4d91-83ac-b90dca63f0c2
md" If you have problems installing ACE1.jl, follow [these steps](https://acesuit.github.io/ACE1docs.jl/dev/gettingstarted/installation/) in Julia's REPL."

# ╔═╡ 97a0b3d1-4411-425b-88b5-68a06faa1329
Pkg.status("InteratomicBasisPotentials")

# ╔═╡ e50d67d9-a00c-4f6f-94f4-d0e4e7b88103
md"### Estimation of the SNAP coefficients β"

# ╔═╡ 8ca8732d-84a5-466d-875b-048b46ecbf82
md"
The choice of the coefficients ``\boldsymbol{\beta}=(\beta_0^1, \tilde{\beta}^1, \dots, \beta_0^l, \tilde{\beta}^l)``
is based on a system of linear equations which considers a large number of atomic
configurations and ``l`` atom types. The matrix formulation for this system ``\mathbf{A \cdot \boldsymbol{\beta}=y}``
is defined in the following equations (see 10.1016/j.jcp.2014.12.018):

```math
\begin{equation*}
   \mathbf{A}=
   \begin{pmatrix}
       \vdots &  &  & & & & & & \\
       N_{s_1} & \sum_{i=1}^{N_{s_1}} B_1^i & \dots & \sum_{i=1}^{N_{s_1}} B_k^i  & \dots & N_{s_L} & \sum_{i=1}^{N_{s_L}} B_1^i & \dots & \sum_{i=1}^{N_{s_L}} B_k^i\\
       \vdots &  &  & & & & & & \\
       0 & -\sum_{i=1}^{N_{s_1}} \frac{\partial B_1^i}{\partial r_j^{\alpha}} & \dots & -\sum_{i=1}^{N_{s_1}} \frac{\partial B_k^i}{\partial r_j^{\alpha}} & \dots &  0 & -\sum_{i=1}^{N_{s_l}} \frac{\partial B_1^i}{\partial r_j^{\alpha}} & \dots & -\sum_{i=1}^{N_{s_l}} \frac{\partial B_k^i}{\partial r_j^{\alpha}} \\
       \vdots &  &  & & & & & & \\
       0 & - \sum_{j=1}^{N_{s_1}} r^j_{\alpha} \sum_{i=1}^{N_{s_1}} \frac{\partial B_1^i}{\partial r_j^{\beta}} & \dots & - \sum_{j=1}^{N_{s_1}} r^j_{\alpha} \sum_{i=1}^{N_{s_1}} \frac{\partial B_k^i}{\partial r_j^{\beta}} & \dots & 0 & - \sum_{j=1}^{N_{s_l}} r^j_{\alpha} \sum_{i=1}^{N_{s_l}} \frac{\partial B_1^i}{\partial r_j^{\beta}} & \dots & - \sum_{j=1}^{N_{s_l}} r^j_{\alpha} \sum_{i=1}^{N_{s_l}} \frac{\partial B_k^i}{\partial r_j^{\beta}}\\
       \vdots &  &  & & & & & & \\
   \end{pmatrix}
\end{equation*}
```

The indexes ``\alpha, \beta = 1,2,3`` depict the ``x``, ``y`` and ``z``
spatial component, ``j`` is an individual atom, and ``s`` a particular configuration.
All atoms in each configuration are considered. The number of atoms of type $m$ in the configuration
``s`` is ``N_{s_m}``.
"

# ╔═╡ 05dc4c56-5298-4c85-ac01-3c7f985439fa
md"
The RHS of the linear system is computed as:

```math
\begin{equation*}
 \mathbf{y}=  \begin{pmatrix}
   \vdots \\
  E^s_{\rm qm} -  E^s_{\rm ref} \\
  \vdots \\\\
  F^{s,j,\alpha}_{\rm qm} - F^{s,j,\alpha}_{\rm ref} \\
  \vdots \\
  W_{\rm qm}^{s,\alpha,\beta} - W_{\rm ref}^{s,\alpha,\beta} \\
  \vdots \\        
   \end{pmatrix}
\end{equation*}
```
"

# ╔═╡ b144a894-fb60-4918-8368-11c8ab6586e8
md"### Loading curated Lennard-Jones cluster data"

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
                position = @SVector [ parse(Float64, line[2]),
					                  parse(Float64, line[3]),
					                  parse(Float64, line[4]) ]
                position += bias
                atoms[i] = Atom(element, position * 1u"Å") 

                force[i] = @SVector [ parse(Float64, line[5]),
					                  parse(Float64, line[6]),
					                  parse(Float64, line[7])]
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

# ╔═╡ 51e181d5-046c-4364-bba6-e378396aad3a
md"### Define SNAP (non-trainable) parameters"

# ╔═╡ 9d42b0ae-24bc-47c2-9481-c1aec9e819e8
begin
	n_atoms = 13
	twojmax = 6
	species = [:Ar]
	rcutfac = 4.0
	rmin0 = 0.0
	rfac0 = 0.99363
	radii = [4.0]
	weight = [1.0]
	snap_params = SNAPParams( n_atoms, twojmax, species, rcutfac,
		                      rmin0, rfac0, radii, weight)
end

# ╔═╡ f4a78391-1d44-4d8a-b20f-92289084b44c
md"### Calculate $A$"

# ╔═╡ 790f9b82-33cb-492c-be5d-14bea9b33335
begin
	train_descriptors_e = [evaluate_basis(sys, snap_params)
		                   for sys in train_systems]
	train_descriptors_f = [evaluate_basis_d(sys, snap_params)
		                   for sys in train_systems]
end

# ╔═╡ 615be86e-d5ac-4317-985f-c56d3d51a3f9
#A = [hcat(train_descriptors_e...)'; hcat(hcat(train_descriptors_f...)...)']

# ╔═╡ b0ff93d9-1a03-4dfd-be26-5008f7fa7d83
A = hcat(train_descriptors_e...)'

# ╔═╡ 45db6c14-26f5-4f4c-a8d4-a39ee2f20258
md"### Calculate $y$"

# ╔═╡ 466fa10f-8598-414e-904f-5a89f1c583d0
#train_forces
y = train_energies

# ╔═╡ b0d40057-6e2c-4aa0-bece-45110fa8138f
md"### Solve ``\mathbf{A \cdot \boldsymbol{\beta}=y}``"

# ╔═╡ eb3fefe7-c682-406b-82fa-85f62ffa3dbc
β = A \ y

# ╔═╡ ca2db5fc-3939-4b05-aa6c-f319283d9ce6
md"### Create an SNAP instance"

# ╔═╡ 6d56e2a6-20c1-4b7b-bd26-fda7660e7b5e
snap = SNAP(β, snap_params)

# ╔═╡ 43afde32-d282-4b85-8ffb-78b9202e8d2d
md"### Test energies"

# ╔═╡ 695a538c-b5c9-4122-899b-b0e716c3eb1c
md"
The local energy can be decomposed into separate contributions for each atom.
SNAP energy can be written in terms of the bispectrum components of the atoms
and a set of coefficients. ``K`` components of the bispectrum are considered
so that ``\mathbf{B}^{i}=\{ B^i_1, \dots, B_K^i\}`` for each atom ``i``, whose
SNAP energy is computed as follows:
```math
   E^i_{\rm SNAP}(\mathbf{B}^i) = \beta_0^{\alpha_i} + \sum_{k=1}^K \beta_k^{\alpha_i} B_k^i =  \beta_0^{\alpha_i} + \boldsymbol{\tilde{\beta}}^{\alpha_i} \cdot \mathbf{B}^i
```
where $\alpha_i$ depends on the atom type.
"

# ╔═╡ 37190555-72bf-47d2-bb11-93416d3dcc40
md"Comparison between the potential energy calculated based on the DFT data, and
the SNAP potential energy calculated based on the bispectrum components
provided by ``snap.jl`` and the fitted coefficients ``\mathbf{\beta}``."

# ╔═╡ b3aacf53-819e-471e-95c3-4a51a44ce391
begin
	fitted_energies = potential_energy.(test_systems, [snap])
	perc_error = 100abs.((test_energies .- fitted_energies) ./ test_energies)
	energy_tests = DataFrame(["Potential Energy"=>test_energies,
		                      "Fitted Potential Energy"=>fitted_energies,
		                      "Percentage Error"=>perc_error])
end

# ╔═╡ 0b22b045-5e14-4bb9-8fb4-45921b287990
md"Root Mean Square Error (RMSE) for energies"

# ╔═╡ d4cfc47b-9d9a-4cd3-9476-1a6a1b722a3e
energy_rmse = sqrt(sum((test_energies .- fitted_energies).^2)/length(test_energies))

# ╔═╡ cf2d9f17-a866-4f4b-b7ed-a4b7986c0a61
md"### Test forces"

# ╔═╡ 9624938e-3709-4b7f-89c5-467aeed8ae79
md"A simple loss function is defined. It calculates the root mean square error (rmse) between each component of the test forces and the forces computed using the SNAP. The arguments of this loss function are forces associated to a particular atomic configuration system of the test data set, N is the system size."

# ╔═╡ ff581378-2145-4041-9cfc-c0ac9ce7eab9
md"$loss(f^{fitted}, f^{test}) = \sqrt{\frac{ {\sum_{i}^{N}\sum_{k=\{x,y,z\}} (f^{fitted}_{i,k} - f^{test}_{i,k}) ^2} } {3N}}$"

# ╔═╡ 086b1812-7df9-4e0f-8749-ce30b0c0a3db
force_rmse(fitted_forces, forces) = 
 sqrt(sum(sum([fdiff.^2 for fdiff in (fitted_forces - forces) ]))/(3.0length(forces)))

# ╔═╡ 57b689c6-7b18-4951-8671-3eea8d5f9baa
begin
	fitted_forces = force.(test_systems, [snap])
	error = force_rmse.(fitted_forces, test_forces)
	force_tests = DataFrame(["Forces"=>test_forces,
		                     "Fitted Forces"=>fitted_forces,
		                     "RMSE"=>error])
end

# ╔═╡ Cell order:
# ╟─e2fb6a18-3da5-497e-9787-bb4af2647b3f
# ╟─99bab0a6-5bbb-4c48-9ff6-1444abd3a6dc
# ╟─73ae3300-fbb3-4a7d-a055-3f995ad55071
# ╠═1247fb54-dbe8-4721-958b-5d615ce24032
# ╠═c7c5a782-2d3e-4409-aff0-f859c4242597
# ╟─a5ef415f-b3a3-4d91-83ac-b90dca63f0c2
# ╠═7d9b8895-0dd4-4b8f-9b15-b45044f8956d
# ╠═97a0b3d1-4411-425b-88b5-68a06faa1329
# ╟─e50d67d9-a00c-4f6f-94f4-d0e4e7b88103
# ╟─8ca8732d-84a5-466d-875b-048b46ecbf82
# ╟─05dc4c56-5298-4c85-ac01-3c7f985439fa
# ╟─b144a894-fb60-4918-8368-11c8ab6586e8
# ╠═311cc38a-8b9a-11ec-362f-edac4da039d4
# ╠═6ea8ce57-5944-4e20-a716-0c71f8f0ad42
# ╟─ca9582f1-d10d-466d-a809-e7439521ea78
# ╠═fb794e2f-1489-4574-b014-a58d6afce3c3
# ╟─51e181d5-046c-4364-bba6-e378396aad3a
# ╠═9d42b0ae-24bc-47c2-9481-c1aec9e819e8
# ╟─f4a78391-1d44-4d8a-b20f-92289084b44c
# ╠═790f9b82-33cb-492c-be5d-14bea9b33335
# ╠═615be86e-d5ac-4317-985f-c56d3d51a3f9
# ╠═b0ff93d9-1a03-4dfd-be26-5008f7fa7d83
# ╟─45db6c14-26f5-4f4c-a8d4-a39ee2f20258
# ╠═466fa10f-8598-414e-904f-5a89f1c583d0
# ╟─b0d40057-6e2c-4aa0-bece-45110fa8138f
# ╠═eb3fefe7-c682-406b-82fa-85f62ffa3dbc
# ╟─ca2db5fc-3939-4b05-aa6c-f319283d9ce6
# ╠═6d56e2a6-20c1-4b7b-bd26-fda7660e7b5e
# ╟─43afde32-d282-4b85-8ffb-78b9202e8d2d
# ╟─695a538c-b5c9-4122-899b-b0e716c3eb1c
# ╟─37190555-72bf-47d2-bb11-93416d3dcc40
# ╠═b3aacf53-819e-471e-95c3-4a51a44ce391
# ╟─0b22b045-5e14-4bb9-8fb4-45921b287990
# ╠═d4cfc47b-9d9a-4cd3-9476-1a6a1b722a3e
# ╟─cf2d9f17-a866-4f4b-b7ed-a4b7986c0a61
# ╟─9624938e-3709-4b7f-89c5-467aeed8ae79
# ╟─ff581378-2145-4041-9cfc-c0ac9ce7eab9
# ╠═086b1812-7df9-4e0f-8749-ce30b0c0a3db
# ╠═57b689c6-7b18-4951-8671-3eea8d5f9baa
