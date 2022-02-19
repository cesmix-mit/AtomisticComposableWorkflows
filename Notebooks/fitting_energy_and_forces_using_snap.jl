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
	Pkg.add("JLD")
end

# ╔═╡ c48b3fc4-191e-490c-9560-0d2de5d5bfa6
begin
	Pkg.add("NBodySimulator")
	Pkg.add("Atomistic")
	using NBodySimulator
	using Atomistic
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
	using JLD
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
The choice of the coefficients ``\boldsymbol{\beta}=(\beta_0^1, \beta^1, \dots, \beta_0^l, \beta^l)``
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
md"### Split data into training and testing."

# ╔═╡ fb794e2f-1489-4574-b014-a58d6afce3c3
begin
	train_systems, train_energies, train_forces = 
								systems[1:50], energies[1:50], forces[1:50];
	test_systems, test_energies, test_forces =
								systems[51:end], energies[51:end], forces[51:end];
end

# ╔═╡ 51e181d5-046c-4364-bba6-e378396aad3a
md"### Define SNAP hyper-parameters"

# ╔═╡ 9d42b0ae-24bc-47c2-9481-c1aec9e819e8
begin
	n_atoms = 13
	twojmax = 6
	species = [:Ar]
	rcutfac = 4.0 #1.0
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
train_descriptors_e = [evaluate_basis(sys, snap_params) for sys in train_systems]

# ╔═╡ ec0b465b-456b-46ee-9d41-658f236fd764
#A_energies = [ 13ones(50,1) hcat(train_descriptors_e...)' ]
A_energies = hcat(train_descriptors_e...)'

# ╔═╡ f1b8f622-f058-464f-b244-0d11a19736d6
train_descriptors_f = [evaluate_basis_d(sys, snap_params) for sys in train_systems]

# ╔═╡ 90f1600e-8f4f-4f59-807d-443129033e7d
A_forces = hcat( [ [train_descriptors_f[s][j][k,α] for k in 1:30]
                   for s in 1:50 for j in 1:13 for α in 1:3]...)'
#A_forces = [ zeros(1950,1) hcat( [ [train_descriptors_f[s][m][j,α] for j in 1:30]
#                                    for s in 1:50 for m in 1:13 for α in 1:3]...)' ]

# ╔═╡ b0ff93d9-1a03-4dfd-be26-5008f7fa7d83
A = [A_energies; A_forces]

# ╔═╡ 45db6c14-26f5-4f4c-a8d4-a39ee2f20258
md"### Calculate $y$"

# ╔═╡ 466fa10f-8598-414e-904f-5a89f1c583d0
begin
	y_energies = train_energies
	y_forces = [train_forces[s][j][α] for s in 1:50 for j in 1:13 for α in 1:3]
	y = [y_energies; y_forces]
end

# ╔═╡ b0d40057-6e2c-4aa0-bece-45110fa8138f
md"### Solve ``\mathbf{A \cdot \boldsymbol{\beta}=y}``"

# ╔═╡ eb3fefe7-c682-406b-82fa-85f62ffa3dbc
β = A \ y
#β = A_energies \ y_energies

# ╔═╡ ca2db5fc-3939-4b05-aa6c-f319283d9ce6
md"### Create an SNAP instance"

# ╔═╡ 6d56e2a6-20c1-4b7b-bd26-fda7660e7b5e
snap = SNAP(β, snap_params)
#snap = SNAP(β[2:end], snap_params)

# ╔═╡ 73e98f25-d9bc-4b8b-b35a-bf3dabcb94fe
md"### Saving SNAP"

# ╔═╡ 08e6dbd8-fc09-4644-bd6e-6a299bcc0d93
#save("snap.jld", "snap", snap)

# ╔═╡ 43afde32-d282-4b85-8ffb-78b9202e8d2d
md"### Test energies"

# ╔═╡ 695a538c-b5c9-4122-899b-b0e716c3eb1c
md"
In the case of total energy, the SNAP contribution can be written in terms of the bispectrum components of the atoms

```math
   E_{\rm SNAP}(\mathbf{r}^N) = N \ \beta_0 + \mathbf{\beta} \cdot \sum_{i=1}^N \mathbf{B}^i 
```
where $\mathbf{\beta}$ is the K-vector of SNAP coefficients and $\beta_0$ is the constant energy contribution for each atom. $\mathbf{B}^i$ is the K-vector of bispectrum components for atom i. 
"

# ╔═╡ 37190555-72bf-47d2-bb11-93416d3dcc40
md"Comparison between the potential energy calculated based on the DFT data, and
the SNAP potential energy calculated based on the bispectrum components
provided by ``InteratomicBasisPotentials.jl`` and the fitted coefficients."

# ╔═╡ b3aacf53-819e-471e-95c3-4a51a44ce391
begin
	#fitted_energies = 13β[1] .+ InteratomicPotentials.potential_energy.(test_systems, [snap])
	fitted_energies = InteratomicPotentials.potential_energy.(test_systems, [snap])
	perc_error = 100abs.((test_energies .- fitted_energies) ./ test_energies)
	energy_tests = DataFrame(["Potential Energy"=>test_energies,
		                      "Fitted Potential Energy"=>fitted_energies,
		                      "Percentage Error"=>perc_error])
end

# ╔═╡ 0b22b045-5e14-4bb9-8fb4-45921b287990
md"Root Mean Square Error (RMSE) for energies"

# ╔═╡ d4cfc47b-9d9a-4cd3-9476-1a6a1b722a3e
rmse_energy = sqrt(sum((test_energies .- fitted_energies).^2)/length(test_energies))

# ╔═╡ cf2d9f17-a866-4f4b-b7ed-a4b7986c0a61
md"### Test forces"

# ╔═╡ 8207a7fa-a353-4b1a-8dd7-ab232e710269
md"
The contribution of the SNAP energy to the force on atom j can be written in terms of the derivatives of the bispectrum components w.r.t. $r_j$ , the position of atom j
```math
   F^j_{\rm SNAP}(\mathbf{B}^i) = - \mathbf{\beta} \cdot \sum_{i=1}^N \frac{\partial \mathbf{B}^i}{\partial \mathbf{r}_j}
```
where $F^j_{\rm SNAP}$ is the force on atom j due to the SNAP energy.
"

# ╔═╡ 9624938e-3709-4b7f-89c5-467aeed8ae79
md"The root mean square error (rmse) between each component of the test forces and the forces computed using the SNAP is defined. The arguments of this function are forces associated to a particular atomic configuration system of the test data set, N is the system size."

# ╔═╡ ff581378-2145-4041-9cfc-c0ac9ce7eab9
md"$rmse_{force}(f^{fitted}, f^{test}) = \sqrt{\frac{ {\sum_{j}^{N}\sum_{\alpha=\{x,y,z\}} (f^{fitted}_{j,\alpha} - f^{test}_{j,\alpha}) ^2} } {3N}}$"

# ╔═╡ 086b1812-7df9-4e0f-8749-ce30b0c0a3db
rmse_force(fitted_forces, forces) = 
 sqrt(sum(sum([fdiff.^2 for fdiff in (fitted_forces - forces)]))/(3.0length(forces)))

# ╔═╡ 57b689c6-7b18-4951-8671-3eea8d5f9baa
begin
	fitted_forces = force.(test_systems, [snap])
	error = rmse_force.(fitted_forces, test_forces)
	force_tests = DataFrame(["Forces"=>test_forces,
		                     "Fitted Forces"=>fitted_forces,
		                     "RMSE"=>error])
end

# ╔═╡ d7bad616-d776-48ae-a452-a14b785e78f7
md"# Classical MD: Lennard-Jones vs SNAP"

# ╔═╡ 67db047b-2b5c-4c2b-8ccc-62e43bd4f74a
begin
	N = 864
	element = :Ar
	box_size = 3.47786u"nm"
	reference_temp = 94.4u"K"
	thermostat_prob = 0.1 # this number was chosen arbitrarily
	Δt = 1e-2u"ps"
	eq_steps = 2000
	prod_steps = 5000

	initial_bodies = 
		generate_bodies_in_cell_nodes(N, element, box_size, reference_temp)

	initial_system = 
		FlexibleSystem(initial_bodies,   
			           CubicPeriodicBoundaryConditions(austrip(box_size)))
	
	eq_thermostat = AndersenThermostat(austrip(reference_temp), 
		                               thermostat_prob / austrip(Δt))

	eq_simulator = NBSimulator(Δt, eq_steps, thermostat = eq_thermostat)
end

# ╔═╡ 258cfc53-47df-40c5-b32c-1468c11dd428
function run_md(potential) 
	eq_result = @time simulate(initial_system, eq_simulator, potential)
	prod_simulator = NBSimulator(Δt, prod_steps, t₀ = get_time(eq_result))
	prod_result = @time simulate(get_system(eq_result), prod_simulator, potential)
	return eq_result, prod_result
end

# ╔═╡ a2c65478-ff10-4d12-8be0-5eede8695608
begin
	ϵ = austrip(1.657e-21u"J")
	σ = austrip(0.34u"nm")
	rcutoff = austrip(0.765u"nm")
	potential_lj = LennardJones(ϵ, σ, rcutoff, species) 
end

# ╔═╡ 50546fbb-b1c6-4952-b48f-8f0a994ae7bc
#eq_result_lj, prod_result_lj = run_md(potential_lj)

# ╔═╡ b647ab98-d8f7-4734-8511-a5d36030b21a
begin
	n_atoms_ = N
	twojmax_ = 6
	rcutfac_ = 4.0 # 1.0
	rmin0_ = 0.0
	rfac0_ = 0.99363
	radii_ = [4.0]
	weight_ = [1.0]
	snap_params_ = SNAPParams( n_atoms_, twojmax_, species, rcutfac_,
		                       rmin0_, rfac0_, radii_, weight_)
	snap_ = SNAP(β, snap_params_)
end

# ╔═╡ 46118f8e-a416-4493-bbdb-06119259b04e
InteratomicPotentials.potential_energy(initial_system, potential_lj)

# ╔═╡ 78d5078c-9cf2-4882-9ebf-1cebf58fbbf7
#13β[1] + InteratomicPotentials.potential_energy(initial_system, snap)
InteratomicPotentials.potential_energy(initial_system, snap_)

# ╔═╡ 232fad90-a77f-4b2d-b939-0c3942af1a44
InteratomicPotentials.energy_and_force(initial_system, potential_lj)

# ╔═╡ 1859fb38-41ca-43d8-b43e-61070fea227f
energy_and_force(initial_system, snap_)

# ╔═╡ 61ce2118-f736-4803-9bde-fffbbbf95c6f
eq_result_lp, prod_result_lp = run_md(snap_)

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
# ╠═ec0b465b-456b-46ee-9d41-658f236fd764
# ╠═f1b8f622-f058-464f-b244-0d11a19736d6
# ╠═90f1600e-8f4f-4f59-807d-443129033e7d
# ╠═b0ff93d9-1a03-4dfd-be26-5008f7fa7d83
# ╟─45db6c14-26f5-4f4c-a8d4-a39ee2f20258
# ╠═466fa10f-8598-414e-904f-5a89f1c583d0
# ╟─b0d40057-6e2c-4aa0-bece-45110fa8138f
# ╠═eb3fefe7-c682-406b-82fa-85f62ffa3dbc
# ╟─ca2db5fc-3939-4b05-aa6c-f319283d9ce6
# ╠═6d56e2a6-20c1-4b7b-bd26-fda7660e7b5e
# ╟─73e98f25-d9bc-4b8b-b35a-bf3dabcb94fe
# ╠═08e6dbd8-fc09-4644-bd6e-6a299bcc0d93
# ╟─43afde32-d282-4b85-8ffb-78b9202e8d2d
# ╟─695a538c-b5c9-4122-899b-b0e716c3eb1c
# ╟─37190555-72bf-47d2-bb11-93416d3dcc40
# ╠═b3aacf53-819e-471e-95c3-4a51a44ce391
# ╟─0b22b045-5e14-4bb9-8fb4-45921b287990
# ╠═d4cfc47b-9d9a-4cd3-9476-1a6a1b722a3e
# ╟─cf2d9f17-a866-4f4b-b7ed-a4b7986c0a61
# ╟─8207a7fa-a353-4b1a-8dd7-ab232e710269
# ╟─9624938e-3709-4b7f-89c5-467aeed8ae79
# ╟─ff581378-2145-4041-9cfc-c0ac9ce7eab9
# ╠═086b1812-7df9-4e0f-8749-ce30b0c0a3db
# ╠═57b689c6-7b18-4951-8671-3eea8d5f9baa
# ╟─d7bad616-d776-48ae-a452-a14b785e78f7
# ╠═c48b3fc4-191e-490c-9560-0d2de5d5bfa6
# ╠═67db047b-2b5c-4c2b-8ccc-62e43bd4f74a
# ╠═258cfc53-47df-40c5-b32c-1468c11dd428
# ╠═a2c65478-ff10-4d12-8be0-5eede8695608
# ╠═50546fbb-b1c6-4952-b48f-8f0a994ae7bc
# ╠═b647ab98-d8f7-4734-8511-a5d36030b21a
# ╠═46118f8e-a416-4493-bbdb-06119259b04e
# ╠═78d5078c-9cf2-4882-9ebf-1cebf58fbbf7
# ╠═232fad90-a77f-4b2d-b939-0c3942af1a44
# ╠═1859fb38-41ca-43d8-b43e-61070fea227f
# ╠═61ce2118-f736-4803-9bde-fffbbbf95c6f
