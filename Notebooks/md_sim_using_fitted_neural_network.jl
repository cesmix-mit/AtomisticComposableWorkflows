### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ ecc74779-8044-4f7f-9a5f-ffd782311438
begin
	import Pkg
	Pkg.Registry.add(
		Pkg.RegistrySpec(url = "https://github.com/cesmix-mit/CESMIX.git"))
end

# ╔═╡ 374e0d20-562f-4f58-aa92-c772e68b78a4
begin
	Pkg.add("Unitful")
	Pkg.add("UnitfulAtomic")
    Pkg.add("LinearAlgebra")
	Pkg.add("StaticArrays")
	Pkg.add("Flux")
	Pkg.add("Zygote")
	Pkg.add("BSON")
	Pkg.add("AtomsBase")
	Pkg.add("InteratomicPotentials")
	Pkg.add("NBodySimulator")
	Pkg.add("Atomistic")
	Pkg.add("Plots")
	Pkg.add("PlutoUI")
end

# ╔═╡ 187acfbc-7fe1-11ec-1eb7-3bc962cbe9cf
begin
	using Unitful
	using UnitfulAtomic
	using LinearAlgebra
	using StaticArrays
	using Flux
	using Zygote
	using BSON
	using AtomsBase
	using InteratomicPotentials
	using NBodySimulator
	using Atomistic
	using Plots
	using PlutoUI
end

# ╔═╡ a35fd14d-119a-4113-9dd8-9787d41195fc
md"# [WIP] Classical MD using empirical and learned potentials"

# ╔═╡ 36efeddd-2c74-4ce9-9f4d-b6b9b1a2a1c5
md"## Installing packages and loading modules"

# ╔═╡ bc22c72b-1df4-427b-a8f9-5858e5a1c9f3
md"Adding the local registry of CESMIX."

# ╔═╡ d59b95b2-b036-46a7-a245-62e603bd9a8f
md"Installation of the necessary packages for this notebook."

# ╔═╡ 76c7ee14-8c7a-4898-bf5b-58b81fe8848a
md"Loading the modules."

# ╔═╡ f33deb52-7ec9-4a70-bc36-62739ff6f88e
Pkg.status("InteratomicPotentials")

# ╔═╡ 7877a16c-6204-40cb-9967-25129abbc663
Pkg.status("Atomistic")

# ╔═╡ 485f2c9c-d116-496b-a55b-c165f3cf3219
md"## MD simulation: Argon case study"

# ╔═╡ ee273d30-26eb-4f8e-bb78-13dbfcd6e334
md"[Argon study case](https://github.com/cesmix-mit/Atomistic.jl/blob/main/examples/argon/lj_simulation_external.jl)"

# ╔═╡ e5c40a4a-3329-41a2-a96b-b3a588ad39db
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

end

# ╔═╡ 00d1b774-c7d3-47d8-8087-ce1e52062aa4
function run_md(potential) 
	eq_simulator = NBSimulator(Δt, eq_steps, thermostat = eq_thermostat)
	eq_result = @time simulate(initial_system, eq_simulator, potential)
	prod_simulator = NBSimulator(Δt, prod_steps, t₀ = get_time(eq_result))
	prod_result = @time simulate(get_system(eq_result), prod_simulator, potential)
	return eq_result, prod_result
end

# ╔═╡ 109438c2-63f8-453a-b9fd-4b0576d971b8
md"## MD using an empirical potential"

# ╔═╡ 02bb5274-f1a6-4012-954f-d0c4e9599d62
begin
	# ϵ = 1.0 # austrip(1.657e-21u"J")
	# σ = 1.0 # austrip(0.34u"nm")
	# rcutoff = 2.0 # austrip(0.765u"nm")
	ϵ = austrip(1.657e-21u"J")
	σ = austrip(0.34u"nm")
	rcutoff = austrip(0.765u"nm")
	potential_lj = LennardJones(ϵ, σ, rcutoff) 
end

# ╔═╡ a01611e4-f734-4227-af3c-1c940c0b857a
eq_result_lj, prod_result_lj = run_md(potential_lj)

# ╔═╡ 207f6094-dba3-4dfe-b6fe-7c51475d0288
md"## MD using a learned potential"

# ╔═╡ 59529eca-b7d7-4899-bdf1-ac2539fbc9d2
md"Loading neural network model and trained parameters."

# ╔═╡ 85725889-64fa-4535-af03-ef2435c61cbd
begin
	model = BSON.load("lennard-jones-nn-model.bson", @__MODULE__)[:cpu_model]
	params = BSON.load("lennard-jones-nn-model-params.bson", @__MODULE__)[:cpu_ps]
	Flux.loadparams!(model, params)
end

# ╔═╡ 0c8426c2-95f6-4c33-b872-bfe83cec6d15
md" Extending InteratomicPotentials.jl with the learned potential."

# ╔═╡ 2e1a070e-fcd7-41c7-bacb-0e4ae2f212cc
begin
	struct LearnedPotential <: EmpiricalPotential
		model
		params
		rcutoff
		#potential::LennardJones
	end
	
	potential_lp = LearnedPotential(model, params, rcutoff)

	function InteratomicPotentials.potential_energy(R::AbstractFloat, 
		                                            p::LearnedPotential)
		#return ϵ * first(p.model([R]))
		InteratomicPotentials.potential_energy(R, potential_lj)
	end
end

# ╔═╡ 542f09a3-bb3e-44b1-afef-69f2cb64a7fa
begin
	function tocart(r, ϕ, θ)
		rx = r * cos(θ) * sin(ϕ)
		ry = r * sin(θ) * sin(ϕ)
		rz = r * cos(ϕ)
		return SVector(rx, ry, rz)
	end
	function force_(r)
		# fx = x -> first(model([norm([x, r[2], r[3]])]))
		# fy = y -> first(model([norm([r[1], y, r[3]])]))
		# fz = z -> first(model([norm([r[1], r[2], z])]))
		# return -ϵ * SVector(  first(gradient(fx, r[1])),
		# 					  first(gradient(fy, r[2])),
		# 					  first(gradient(fz, r[3])))
		return force(norm(r), r, potential_lj)
	end
	r0=σ/2; r1=rcutoff; 
	nr = 600; nϕ = 70; nθ = 140
	Δr = (r1 - r0) / (nr - 1); Δϕ = π / (nϕ - 1); Δθ = 2π / (nθ - 1)
	rs = r0:Δr:r1; ϕs = 0.0:Δϕ:π; θs = -π:Δθ:π
	precomputed_forces = Array{Union{Missing, SVector}}(missing, nr, nϕ, nθ)
	#precomputed_forces = [force_(tocart(r, ϕ, θ)) for r in rs, ϕ in ϕs, θ in θs]
end

# ╔═╡ 5985cdf3-f8c9-4d2f-8403-fada817983c2
function InteratomicPotentials.force(R::AbstractFloat,
		                             r::SVector{3,<:AbstractFloat}, 
		                             p::LearnedPotential)
	    rr = sqrt(sum(r.^2))
		θ = atan(r[2], r[1])
		ϕ = acos(r[3]/rr)
		i = trunc(Int, (rr - r0) / Δr + 1)
		j = trunc(Int, ϕ / Δϕ + 1)
		k = trunc(Int, (θ + π)  / Δθ + 1)

		if i < 1
			i = 1
			rr = r0
		elseif i > nr
			i = nr
			rr = r1
		end

		if isequal(precomputed_forces[i, j, k], missing)
			precomputed_forces[i, j, k] = force_(tocart(rr, ϕ, θ))
		end
			
		return precomputed_forces[i, j, k]
end

# ╔═╡ 9f875d26-9b0c-4c5e-bc6e-73b2fd3e4c33
begin
	rss = []; fs_model = []; fs_surr = []
	for r in 2r0:Δr/10:r1
		for _ in 1:30
			ϕ = rand() * 2.0 * π; θ = rand() * π
			x_ij = r * cos(ϕ) * sin(θ) 
			y_ij = r * sin(ϕ) * sin(θ) 
			z_ij = r * cos(θ)          
			r_ij = SVector(x_ij, y_ij, z_ij)
			push!(rss, r)
			push!(fs_model, force(norm(r_ij), r_ij, potential_lp))
			push!(fs_surr, force(norm(r_ij), r_ij, potential_lj))
		end
	end
end

# ╔═╡ 34c6c57e-d930-42ce-99eb-3d253cad8a05
begin
	plot(rss, norm.(fs_model), seriestype = :scatter, markerstrokewidth=0,
		 xlabel = "Norm of the position difference, |ri-rj|", 
	  	 ylabel = "Norm of the force", 
		 label = "Force norm from precomputed matrix")
	plot!(rss, norm.(fs_surr), seriestype = :scatter, markerstrokewidth=0,
		  label = "LJ force norm")
end

# ╔═╡ ca1b7f43-042a-44f9-93b2-ea08cdfe9f41
begin
	px =plot( [f[1] for f in fs_surr], [f[1] for f in fs_model], 
		      seriestype = :scatter, markerstrokewidth=0, label="",
		      xlabel = "Fx, LJ", ylabel = "Fx, precomputed LJ")
	py =plot( [f[2] for f in fs_surr], [f[2] for f in fs_model], 
		      seriestype = :scatter, markerstrokewidth=0, label="",
              xlabel = "Fy, LJ", ylabel = "Fy, precomputed LJ")
    pz =plot( [f[3] for f in fs_surr], [f[3] for f in fs_model], 
		      seriestype = :scatter, markerstrokewidth=0, label="",
              xlabel = "Fz, LJ", ylabel = "Fz, precomputed LJ")
	plot(px, py, pz)
end

# ╔═╡ 1e471b6e-b697-4f0e-a058-a28076800976
eq_result_lp, prod_result_lp = run_md(potential_lp)

# ╔═╡ 9a7ad367-15b0-4430-b24c-7ac656f296c0
sum([ isequal(precomputed_forces[i, j, k], missing) for i in 1:nr, j in 1:nϕ, k in 1:nθ])

# ╔═╡ 5f78419e-3abe-4b17-a331-e40ed88c596c
md"## Comparing results"

# ╔═╡ 5e40f7ed-4070-4743-b8d7-a7f9404d82a4
md"Temperature"

# ╔═╡ b1b35543-65df-4dde-b1f8-80660a0ce13c
begin
	temp_lj = plot_temperature(eq_result_lj, 10)
	plot_temperature!(temp_lj, prod_result_lj, 10)
end

# ╔═╡ 9431472f-2f13-40f2-8835-619be87d8848
begin
	temp_lp = plot_temperature(eq_result_lp, 10)
	plot_temperature!(temp_lp, prod_result_lp, 10)
end

# ╔═╡ 76b1738c-2375-4c30-9ce6-cd28ef4425e8
md"Energy"

# ╔═╡ 4c9eb6bf-a2c0-4b38-952e-782e6e1b24af
begin
	energy_lj = plot_energy(eq_result_lj, 10)
	plot_energy!(energy_lj, prod_result_lj, 10)
end

# ╔═╡ c0f4b3a7-cee6-4530-af14-3742e1c41baf
begin
	energy_lp = plot_energy(eq_result_lp, 10)
	plot_energy!(energy_lp, prod_result_lp, 10)
end

# ╔═╡ 0e171494-0f04-41ec-bff1-99d9ce0935a2
md"Radial distribution function"

# ╔═╡ 31b4e7fe-0807-4316-b628-a403a2d2c1cb
plot_rdf(prod_result_lj, potential_lj.σ, Int(0.95 * prod_steps))

# ╔═╡ 6db35db8-eac3-4dc0-aefe-c0701fc729ef
plot_rdf(prod_result_lp, potential_lj.σ, Int(0.95 * prod_steps))

# ╔═╡ Cell order:
# ╟─a35fd14d-119a-4113-9dd8-9787d41195fc
# ╟─36efeddd-2c74-4ce9-9f4d-b6b9b1a2a1c5
# ╟─bc22c72b-1df4-427b-a8f9-5858e5a1c9f3
# ╠═ecc74779-8044-4f7f-9a5f-ffd782311438
# ╟─d59b95b2-b036-46a7-a245-62e603bd9a8f
# ╠═374e0d20-562f-4f58-aa92-c772e68b78a4
# ╟─76c7ee14-8c7a-4898-bf5b-58b81fe8848a
# ╠═187acfbc-7fe1-11ec-1eb7-3bc962cbe9cf
# ╠═f33deb52-7ec9-4a70-bc36-62739ff6f88e
# ╠═7877a16c-6204-40cb-9967-25129abbc663
# ╟─485f2c9c-d116-496b-a55b-c165f3cf3219
# ╟─ee273d30-26eb-4f8e-bb78-13dbfcd6e334
# ╠═e5c40a4a-3329-41a2-a96b-b3a588ad39db
# ╠═00d1b774-c7d3-47d8-8087-ce1e52062aa4
# ╟─109438c2-63f8-453a-b9fd-4b0576d971b8
# ╠═02bb5274-f1a6-4012-954f-d0c4e9599d62
# ╠═a01611e4-f734-4227-af3c-1c940c0b857a
# ╟─207f6094-dba3-4dfe-b6fe-7c51475d0288
# ╟─59529eca-b7d7-4899-bdf1-ac2539fbc9d2
# ╠═85725889-64fa-4535-af03-ef2435c61cbd
# ╟─0c8426c2-95f6-4c33-b872-bfe83cec6d15
# ╠═2e1a070e-fcd7-41c7-bacb-0e4ae2f212cc
# ╠═542f09a3-bb3e-44b1-afef-69f2cb64a7fa
# ╠═5985cdf3-f8c9-4d2f-8403-fada817983c2
# ╠═9f875d26-9b0c-4c5e-bc6e-73b2fd3e4c33
# ╠═34c6c57e-d930-42ce-99eb-3d253cad8a05
# ╠═ca1b7f43-042a-44f9-93b2-ea08cdfe9f41
# ╠═1e471b6e-b697-4f0e-a058-a28076800976
# ╠═9a7ad367-15b0-4430-b24c-7ac656f296c0
# ╟─5f78419e-3abe-4b17-a331-e40ed88c596c
# ╟─5e40f7ed-4070-4743-b8d7-a7f9404d82a4
# ╠═b1b35543-65df-4dde-b1f8-80660a0ce13c
# ╠═9431472f-2f13-40f2-8835-619be87d8848
# ╟─76b1738c-2375-4c30-9ce6-cd28ef4425e8
# ╠═4c9eb6bf-a2c0-4b38-952e-782e6e1b24af
# ╠═c0f4b3a7-cee6-4530-af14-3742e1c41baf
# ╟─0e171494-0f04-41ec-bff1-99d9ce0935a2
# ╠═31b4e7fe-0807-4316-b628-a403a2d2c1cb
# ╠═6db35db8-eac3-4dc0-aefe-c0701fc729ef
