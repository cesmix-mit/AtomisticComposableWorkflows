### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ e7977ff7-5bf3-4d2d-b8e8-d4e3c760c801
begin
	import Pkg
	Pkg.Registry.add(
		Pkg.RegistrySpec(url = "https://github.com/cesmix-mit/CESMIX.git"))
end

# ╔═╡ 0fc66123-e952-4bdc-941a-4777eaab867a
begin
	Pkg.add("StaticArrays")
	Pkg.add("LinearAlgebra")
	Pkg.add("InteratomicPotentials")
	Pkg.add("Flux")
	Pkg.add("Zygote")
	Pkg.add("BSON")
	Pkg.add("Plots")
	Pkg.add("PlutoUI")
end

# ╔═╡ 8afc5295-ff66-4e3d-bdb9-7c1521ce0063
begin
	using StaticArrays
	using LinearAlgebra
	using InteratomicPotentials
	using Flux
	using Zygote
	using BSON
	using Plots
	using PlutoUI
end

# ╔═╡ f632db9e-3e64-4c60-9870-7aa187ae8577
md"# Create your own potential"

# ╔═╡ 55919ec4-d5dd-494b-9a26-535634cd9f17
md"##### Installing packages and loading modules"

# ╔═╡ 0881ffb7-5f56-43b9-a6eb-b67b3eebffb4
md"##### Define the potential type"

# ╔═╡ 74d1882b-fcac-41c6-ba0b-aa4b4cf47ae9
struct NNPotential <: EmpiricalPotential
  model
  params
  rcutoff
end

# ╔═╡ bb2cc1c2-2eec-4cc6-bd4f-c46005259ec2
md"##### Define how to calculate the energy"

# ╔═╡ 1ad287cb-541c-48fa-b857-0ba3b210eb9a
function InteratomicPotentials.potential_energy(R::AbstractFloat, 
	                                            p::NNPotential)
	  return first(p.model([R]))
end

# ╔═╡ e8822845-191d-45df-9e36-234c13034a1c
md"##### Define how to calculate the force using AD"

# ╔═╡ 27497066-74e6-451d-9dc5-0c16ce24dc27
md"##### Load neural network model and parameters"

# ╔═╡ 8242abf7-d6a2-47fe-8510-23d995d74692
begin
	model = BSON.load("lennard-jones-nn-model.bson", @__MODULE__)[:cpu_model]
	params = BSON.load("lennard-jones-nn-model-params.bson", @__MODULE__)[:cpu_ps]
	Flux.loadparams!(model, params)
end

# ╔═╡ fbf032b2-968b-11ec-1450-9b086549e23d
function InteratomicPotentials.force(R::AbstractFloat,
									 r::SVector{3,<:AbstractFloat}, 
									 p::NNPotential)
  fx = x -> first(model([norm([x, r[2], r[3]])]))
  fy = y -> first(model([norm([r[1], y, r[3]])]))
  fz = z -> first(model([norm([r[1], r[2], z])]))
  return  -SVector( first(gradient(fx, r[1])),
					first(gradient(fy, r[2])),
					first(gradient(fz, r[3])))
end

# ╔═╡ cad4a56e-a5de-434a-8081-8b899f168c09
md"##### Define your potential"

# ╔═╡ 03055766-24b5-41c3-9e04-1145dafe3550
begin
	rcutoff = 14.45
	p = NNPotential(model, params, rcutoff)
end

# ╔═╡ f2aa56dd-a79c-4ef8-9fc1-bb802f71bac4
md"##### Compute energy and force with the new potential"

# ╔═╡ d5e7e10e-a711-486e-8501-30cf94c66630
r_diff = SVector(0.5, 0.5, 0.5); r_diff_n = norm(r_diff)

# ╔═╡ 1ee2c42b-72ee-4e65-ba43-1a1b0385ab9b
potential_energy(r_diff_n, p)

# ╔═╡ 02e183a4-ba6f-4610-9c08-f004375bd706
force(r_diff_n, r_diff, p)

# ╔═╡ 3cb2259a-e24f-4ec9-b29e-6c4b8569b09d
md"##### Ask Julia how many different \"potential_energy\" methods it has."

# ╔═╡ 659c820d-bf90-4bad-bf0a-7ebf28faef32
methods(potential_energy)

# ╔═╡ Cell order:
# ╟─f632db9e-3e64-4c60-9870-7aa187ae8577
# ╟─55919ec4-d5dd-494b-9a26-535634cd9f17
# ╠═e7977ff7-5bf3-4d2d-b8e8-d4e3c760c801
# ╠═0fc66123-e952-4bdc-941a-4777eaab867a
# ╠═8afc5295-ff66-4e3d-bdb9-7c1521ce0063
# ╟─0881ffb7-5f56-43b9-a6eb-b67b3eebffb4
# ╠═74d1882b-fcac-41c6-ba0b-aa4b4cf47ae9
# ╟─bb2cc1c2-2eec-4cc6-bd4f-c46005259ec2
# ╠═1ad287cb-541c-48fa-b857-0ba3b210eb9a
# ╟─e8822845-191d-45df-9e36-234c13034a1c
# ╠═fbf032b2-968b-11ec-1450-9b086549e23d
# ╟─27497066-74e6-451d-9dc5-0c16ce24dc27
# ╠═8242abf7-d6a2-47fe-8510-23d995d74692
# ╟─cad4a56e-a5de-434a-8081-8b899f168c09
# ╠═03055766-24b5-41c3-9e04-1145dafe3550
# ╟─f2aa56dd-a79c-4ef8-9fc1-bb802f71bac4
# ╠═d5e7e10e-a711-486e-8501-30cf94c66630
# ╠═1ee2c42b-72ee-4e65-ba43-1a1b0385ab9b
# ╠═02e183a4-ba6f-4610-9c08-f004375bd706
# ╟─3cb2259a-e24f-4ec9-b29e-6c4b8569b09d
# ╠═659c820d-bf90-4bad-bf0a-7ebf28faef32
