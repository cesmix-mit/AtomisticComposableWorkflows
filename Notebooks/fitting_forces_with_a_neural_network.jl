### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ a904c8dc-c7fa-4ef7-97f0-9018bb9f2db2
begin
import Pkg
Pkg.add("Unitful")
Pkg.add("PeriodicTable")
Pkg.add("StaticArrays")
Pkg.add("LinearAlgebra")
Pkg.add("AtomsBase")
Pkg.add("InteratomicPotentials")
#Pkg.add(url="https://github.com/JuliaMolSim/AtomsBase.jl", rev="v0.1.0")
#Pkg.add(url="https://github.com/cesmix-mit/InteratomicPotentials.jl", rev="v0.1.2")
end

# ╔═╡ 5f6d4c16-bf9f-442a-bb43-5cbe1b861fb1
begin
using StaticArrays
using LinearAlgebra
using Unitful
using PeriodicTable
using AtomsBase
using InteratomicPotentials
using Flux
using Flux.Data: DataLoader
using CUDA
using Plots
using PlutoUI
end

# ╔═╡ 5dd8b98b-967c-46aa-9932-25157a10d0c2
md" # Fitting atomic forces with a neural network using Julia"

# ╔═╡ 09985247-c963-4652-9715-1f437a07ef81
md"As part of the [CESMIX](https://computing.mit.edu/cesmix/) project, novel Julia tools are being developed to perform large-scale material simulations. In this context, this notebook presents a simple example of how to fit interatomic forces with a neural network.  In particular, the following points are addressed:

1) Generate simple surrogate DFT data
2) Define a loss function and a neural network model
3) Train the model with the DFT data

Let's do it!
"

# ╔═╡ 3bdb681d-f9dc-4d37-8667-83fccc247b3d
md"## Installing packages and loading modules"

# ╔═╡ e0e8d440-1df0-4581-8277-d3d6886351d7
md" What are these packages needed for?

- StaticArrays is used to define small vectors such as forces or velocities. The speed of small SVectors, SMatrixs and SArrays is often > 10 × faster than Base.Array.
- LinearAlgebra is used to compute operations such as norms.
- Unitful is used to associate physical units to parameters and variables, and to verify that their operations are correct.
- PeriodicTable is used to access information about the elements of the periodic table. In this notebook Argon is used.
- AtomsBase is used as an abstract interface for representation of atomic geometries. 
- InteratomicPotentials is used to computes methods (energies, forces, and virial tensors) for a variety of interatomic potentials.
- Flux is used to define and train the neural network model.
- CUDA is used to parallelize the training and execution of the neural network model.
"

# ╔═╡ 29e753a9-f49d-41b4-85f3-e8f35d2d67f5
md"The following block install required packages"

# ╔═╡ 6a5f238c-0672-4fcb-9da9-3c2f5a718d5a
md"The next block allows to load the modules"

# ╔═╡ b562c28b-a530-41f8-b2d4-57e2df2860c5
md" ## Generating a simple surrogate DFT dataset"

# ╔═╡ 7bcb0360-a656-4637-b0ce-e4ada1e9ce0a
md"The following function generates a vector of atomic configurations. Each atomic configuration is defined using AtomsBase.jl, in particular with a `FlexibleSystem`. It contains information about the domain and its boundaries, as well as the atoms that compose it. The domain has zero Dirichlet boundary conditions. In this example each configuration contains a binary Argon system. Each atom is represented by a `StaticAtom`. The positions of the atoms are calculated randomly within the domain under the constraint that both atoms are at a fixed distance. Modify this function if you want to change the distribution of the atoms :)"

# ╔═╡ 3e606585-86b6-4389-818d-bcbdb6078608
function gen_atomic_confs()
    # Domain
    L = 10u"Å"
    box = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] * L
    # Boundary conditions
    bcs = [DirichletZero(), DirichletZero(), DirichletZero()]
    # No. of atoms per configuration
    N = 2
    # No. of configurations
    M = 50000
    # Element
    elem = elements[:Ar]
    # Define atomic configurations
    atomic_confs = []
	dists = collect(1.15:0.001:2.1)u"Å"
	for j in 1:M
		d = dists[rand(1:length(dists))]
        atoms = []
        x1 = rand(0.1:0.1:L.val)u"Å"
        y1 = rand(0.1:0.1:L.val)u"Å"
        z1 = rand(0.1:0.1:L.val)u"Å"
        pos1 = SVector{3}(x1, y1, z1)
        atom1 = StaticAtom(pos1, elem)
        push!(atoms, atom1)
		x2 = L; y2 = L; z2 = L
		while x1 + x2 > L       || y1 + y2 > L    || z1 + z2 > L  ||
		      x1 + x2 < 0.0u"Å" || y1 + y2 < 0.0u"Å" || z1 + z2 < 0.0u"Å"
        	ϕ = rand() * 2.0 * π; θ = rand() * π
	        x2 = d * cos(ϕ) * sin(θ)
	        y2 = d * sin(ϕ) * sin(θ)
	        z2 = d * cos(θ)
		end
        pos2 = SVector{3}(x1+x2, y1+y2, z1+z2)
        atom2 = StaticAtom(pos2, elem)
        push!(atoms, atom2)
        push!(atomic_confs, FlexibleSystem(box, bcs, atoms))
    end
    return atomic_confs
end

# ╔═╡ 19dc1efa-27ac-4c94-8f4d-bc7886c2959e
md"To fit the data a loss function will be defined. This function requires to precompute information about the neighbors of each atom. In particular, the position difference, $r_{i,j} = r_i - r_j$, between the atom $i$ to each of its neighbors $j$ is precomputed."

# ╔═╡ c0d477c3-3af8-4b19-bf84-657d4c60fea8
md"$neighbors\_pos\_diffs(i) = \{ r_{i,j} \ where \ j \  \epsilon \ neighbors(i)  \}$"

# ╔═╡ 4430121b-9d15-4c12-86f8-48fa1579845b
md"The neighbors $j$ of the atom $i$ are those within a radius $r_{cut}$ around it:"

# ╔═╡ 160f3fb3-8dbd-4a3b-befe-1bef361fdd69
md"$neighbors(i) = \{ j \neq i \  / \ |r_i - r_j| < r_{cut} \}$"

# ╔═╡ 67d9544a-4ff4-4703-8579-b8a335d58f63
function compute_neighbor_pos_diffs(atomic_conf, rcutoff)
    neighbor_pos_diffs = []
    N = length(atomic_conf)
    for i in 1:N
        neighbor_pos_diffs_i = []
        for j in 1:N
            rij = position(getindex(atomic_conf, i)) -
                  position(getindex(atomic_conf, j))
			rij = [map((x -> x.val), rij.data)...] # removing units
            if 0 < norm(rij) < rcutoff
                push!(neighbor_pos_diffs_i, rij)
            end
        end
        push!(neighbor_pos_diffs, neighbor_pos_diffs_i)
    end
    return neighbor_pos_diffs
end

# ╔═╡ 0fa4ca5a-f732-4630-991c-ff5e4b76c536
md"Next function calculates the surrogate DFT force $f^{dft}_i$ of each atom $i$ using the position difference $r_{i,j}$ to each of its neighbors $j$ and the potential $p$ (e.g. [LennardJones](https://en.wikipedia.org/wiki/Lennard-Jones_potential))."

# ╔═╡ 40c9d9cd-05af-4bbf-a772-5c09c1b03a66
md"$f^{dft}_i = \sum_{\substack{j \neq i \\ |r_i - r_j| < r_{cut}}} f^{dft}_{i,j}$"

# ╔═╡ 93450770-74c0-42f7-baca-7f8276373f9f
md"$f^{dft}_{i,j} = - \nabla LJ_{(r_{i,j})} = 24 ϵ  \left( 2 \left( \frac{σ}{|r_{i,j}|} \right) ^{12} -  \left( \frac{σ}{|r_{i,j}|} \right) ^6  \right) \frac{r_{i,j} }{|r_{i,j}|^2 }$"

# ╔═╡ 216f84d3-ec5c-4c49-a9d7-af0e98907e15
md"Here, the force between $i$ and $j$ is computed by the [InteratomicPotentials.jl](https://github.com/cesmix-mit/InteratomicPotentials.jl) library."

# ╔═╡ 6e20dcb0-2fed-4637-ae07-775f3cd4a8dd
function compute_forces(neighbor_pos_diffs_i, p)
    return [ length(rij)>0 ? sum(force.(rij, [p])) : SVector(0.0, 0.0, 0.0)                        for rij in neighbor_pos_diffs_i ]
end

# ╔═╡ 8aa8f56a-24a1-4ddd-840e-91517fd27b9c
md"The following function generates the training and test data sets consisting of the position differences with the neighbors of each atom and the surrogate DFT forces. The data sets are divided into batches using `Flux.DataLoader` and transferred to the GPU if necessary."


# ╔═╡ 8787b0dc-81f5-4e15-8931-4f8325d45061
function gen_data(train_prop, batchsize, rcutoff, p, use_cuda, device)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Generate atomic configurations
    atomic_confs = gen_atomic_confs()

    # Generate neighbor position differences
    neighbor_pos_diffs = vcat(compute_neighbor_pos_diffs.(atomic_confs, rcutoff)...)
    N = floor(Int, train_prop * length(neighbor_pos_diffs))
    train_neighbor_pos_diffs = neighbor_pos_diffs[1:N]
    test_neighbor_pos_diffs = neighbor_pos_diffs[N+1:end]

    # Generate learning data using the potential `p` and the atomic configurations
    f_train = compute_forces(train_neighbor_pos_diffs, p)
    f_test = compute_forces(test_neighbor_pos_diffs, p)
    
    # If CUDA is used, convert SVector to Vector and transfer vectors to GPU
    if use_cuda
        train_neighbor_pos_diffs = device([ device.(convert.(Vector, d)) for d in train_neighbor_pos_diffs ])
		test_neighbor_pos_diffs = device([ device.(convert.(Vector, d)) for d in test_neighbor_pos_diffs ])
        f_train = device.(convert.(Vector, f_train))
        f_test = device.(convert.(Vector, f_test))
    end

    # Create DataLoaders (mini-batch iterators)
    train_loader = DataLoader((train_neighbor_pos_diffs, f_train), batchsize=batchsize, shuffle=true)
    test_loader  = DataLoader((test_neighbor_pos_diffs, f_test), batchsize=batchsize)

    return train_loader, test_loader
end

# ╔═╡ 152f05fb-ac0a-4d8d-8975-94585443853a
md" ## Defining the loss functions"

# ╔═╡ fe7cf55f-cbc9-41db-8554-03fdecbc881d
md"A simple loss function is defined. It calculates the root mean square error (rmse) between each component of the surrogate forces and the forces computed using the neural network model (defined below). The arguments of this loss function are batches of the training or test data set, N is the batch size."

# ╔═╡ dd5e6c1f-ad9b-49fa-8b5e-9cffd5781c16
md"$loss(f^{model}, f^{surr}) = \sqrt{\frac{ {\sum_{i}^{N}\sum_{k=\{x,y,z\}} (f^{model}_{i,k} - f^{surr}_{i,k}) ^2} } {3N}}$"

# ╔═╡ 7b48d757-18eb-4f6b-bf7f-53d8c52cf7a1
loss(f_model, forces) = 
    sqrt(sum(sum([fdiff.^2 for fdiff in (f_model - forces) ])) / (3.0*length(forces)))

# ╔═╡ f93f298d-6ac0-45e2-abee-ab36d85b3f7b
md"The force of the atom $i$ is computed as"

# ╔═╡ 1180e03c-482e-428a-b66c-5534bcb4de83
md"$f^{model}_i = \sum_{r_{i,j} \ \epsilon \ neighbors\_dists(i) } model(r_{i,j})$"

# ╔═╡ 08fbcf33-0935-41ce-a721-c597e2570b47
md"The following function computes $f^{model}$ for a batch of atoms"

# ╔═╡ 8a03ffbc-9ab2-438d-b71b-1d7238b33507
f_model(neighbor_pos_diffs_i, model) = length(neighbor_pos_diffs_i)>0 ?
                                       sum(model.(neighbor_pos_diffs_i)) : zeros(3)

# ╔═╡ 2c629ed0-e21d-4f38-b46f-fc2a1757b62f
md"The global loss or loss of an entire data set (training or test) is the average of the losses of the batches in that data set. M is the number of batches."

# ╔═╡ 48e7fa16-ea5f-44f5-84ab-2994e7382c4e
md"$\frac{1}{M} \sum_{b \ \epsilon \ \#batches(training\_set)} loss(f^{model}_b, f^{surr}_b)$"

# ╔═╡ 7410e309-2ee0-4b65-af2a-c4f738a3bd80
global_loss(loader, model) = 
        sum([loss(f_model.(d, [model]), f) for (d, f) in loader]) / length(loader)

# ╔═╡ 2244cd54-f927-40e0-8fad-75a34096044d
md"More complex examples can also fit energies and tensors, and define hybrid potentials."

# ╔═╡ b481e2f6-1a87-4af3-8d93-111e2ab8d933
md"## Defining and training the model in CPU"

# ╔═╡ 01de26b8-c596-4690-bb62-07170260ddbc
md"The first step is to define an interatomic potential to generate the surrogate forces. In particular, the Lennard-Jones potential is defined using [InteratomicPotentials.jl](https://github.com/cesmix-mit/InteratomicPotentials.jl)"

# ╔═╡ 93e7f0c4-ad8c-473e-a8d3-09508f2ba6df
begin
lj_ϵ = 1.0
lj_σ = 1.0
lj = LennardJones(lj_ϵ, lj_σ);
end

# ╔═╡ f1a29169-2c8a-41bc-9c35-be2c77b418ec
md"Training and test datasets are generated based on the interatomic potential, the radius of the neighborhoods ($rcutoff$), the training size proportion ($train\_prop$), the batch size ($batchsize$), and parameters related to CPU/GPU usage ($use\_cuda$ and $device1$). "

# ╔═╡ 54891249-99be-4798-a0e9-5b0193ecaa3c
begin
rcutoff = 2.5 * lj_σ
train_prop = 0.8
batchsize = 256
use_cuda = false
device1 = cpu
cpu_train_loader, cpu_test_loader = gen_data(train_prop, batchsize, rcutoff, lj, use_cuda, device1)
end

# ╔═╡ 1cb8b6d0-d645-4d16-a501-cc9d2a047b39
md"The neural network model and the parameters to be optimized are defined using [Flux.jl](https://github.com/FluxML/Flux.jl). An explanation about `Chain` and `Dense` is presented [here](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Chain)."

# ╔═╡ c8c43e19-daff-4272-8703-a2dcaceca7a9
begin
cpu_model = Chain(Dense(3,200,Flux.relu),Dense(200,200,Flux.relu),Dense(200,200,Flux.relu),Dense(200,3))
cpu_ps = Flux.params(cpu_model) # model's trainable parameters
end

# ╔═╡ fcccc8ca-97c8-43cd-ab1d-b76f126ab617
md"The optimizer [ADAM](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) is defined using a learning rate η = 0.001."

# ╔═╡ 49d7f00a-db88-4dab-9209-c9840255d999
opt = ADAM(0.001)

# ╔═╡ 4481fcae-6c0c-4455-9a11-0eb1f19805cc
md"The model is trained in the loop below. At each iteration or epoch, the gradient of the loss function is calculated for each batch of the training set. Flux then uses this gradient and the optimizer to update the parameters."

# ╔═╡ 2e199532-5177-4b39-9c7e-83d61c1e2f13
begin
epochs = 15
with_terminal() do
	for epoch in 1:epochs
	    # Training of one epoch
	    time = Base.@elapsed for (d, f) in cpu_train_loader
	        gs = gradient(() -> loss(f_model.(d, [cpu_model]), f), cpu_ps)
	        Flux.Optimise.update!(opt, cpu_ps, gs)
	    end
	    
	    # Report traning loss
	    println("Epoch: $(epoch), loss: $(global_loss(cpu_train_loader, cpu_model)), time: $(time)")
	end
end
end

# ╔═╡ 74d94c45-7a66-4779-9bd8-d6987d23bdc4
md"### Test CPU results"

# ╔═╡ 46a7f1ab-5f99-4b69-9bff-c299815cccab
md"The loss of the test data set is calculated using the root mean squared error (rmse)"

# ╔═╡ 195925f3-19ea-47a2-bc31-561bf698f580
@show "Test RMSE: $(global_loss(cpu_test_loader, cpu_model))"

# ╔═╡ 97cd889b-7fd5-494a-8f34-5830e53557f6
md"The following plots show the forces computed by the neural network model and the surrogate model w.r.t. the norm of the position differences $|r_i-r_j|$. 30 forces are computed per each $|r_i-r_j|$."

# ╔═╡ abf7c48c-221b-43d7-a583-716b192bc5f6
begin
	rs = 1.15:0.001:2.0
	rss = []
	fs_model = []
	fs_surr = []
	for r in rs
		for _ in 1:30
			ϕ = rand() * 2.0 * π; θ = rand() * π
			x_ij = r * cos(ϕ) * sin(θ) 
			y_ij = r * sin(ϕ) * sin(θ) 
			z_ij = r * cos(θ)          
			r_ij = [x_ij, y_ij, z_ij]
			push!(rss, r)
			push!(fs_model, cpu_model(r_ij))
			push!(fs_surr, force(r_ij, lj))
		end
	end
end

# ╔═╡ 276e1787-d15e-4094-bf00-30b6192228e1
begin
	plot(rss, norm.(fs_model), seriestype = :scatter, markerstrokewidth=0,
		 xlabel = "Norm of the position difference, |ri-rj|", 
	  	 ylabel = "Norm of the force", 
		 label = "Neural network model force norm")
	plot!(rss, norm.(fs_surr), seriestype = :scatter, markerstrokewidth=0,
		  label = "Surrogate force norm")
end

# ╔═╡ d22ab6ee-4184-4dfc-a9bb-2af4f0de5882
begin
	px =plot( [f[1] for f in fs_surr], [f[1] for f in fs_model], 
		      seriestype = :scatter, markerstrokewidth=0, label="",
		      xlabel = "Fx, surrogate model", ylabel = "Fx, NN model")
	py =plot( [f[2] for f in fs_surr], [f[2] for f in fs_model], 
		      seriestype = :scatter, markerstrokewidth=0, label="",
              xlabel = "Fy, surrogate model", ylabel = "Fy, NN model")
    pz =plot( [f[3] for f in fs_surr], [f[3] for f in fs_model], 
		      seriestype = :scatter, markerstrokewidth=0, label="",
              xlabel = "Fz, surrogate model", ylabel = "Fz, NN model")
	plot(px, py, pz)
end

# ╔═╡ d720916f-8ed8-4caf-bcf1-43da2f78e21a
md"Test invariants with respect to permutation, translation, and rotation."

# ╔═╡ 4d94701b-e3a5-4aef-ba23-97c1150f89a1
md"## Defining and training the model in GPU"

# ╔═╡ c2492af2-23e7-45c5-bf44-109c59d7986d
md"This case is analogous to that of the CPU except for some differences. First, when training and test data sets are generated, they must be transferred to the GPU. This is done in the `gen_data` function."

# ╔═╡ d4526269-07e6-4684-8bed-7cee5898ef52
begin
use_cuda_ = true
device2 = gpu
gpu_train_loader, gpu_test_loader = gen_data(train_prop, batchsize, rcutoff, lj, use_cuda_, device2)
end

# ╔═╡ a67f8826-ee85-4e74-aec5-dd2c79c56c13
md"Also, the model with its parameters is transferred to the GPU."

# ╔═╡ 5a7e8a3f-3d8d-4048-80fe-c64e9c1cfd44
begin
gpu_model = Chain(Dense(3,200,Flux.relu),Dense(200,200,Flux.relu),Dense(200,3)) |> device2
gpu_ps = Flux.params(gpu_model) # model's trainable parameters
end

# ╔═╡ f89622bb-433a-4547-aec0-9fd3a2ffbaf2
md"The model is trained in the loop below. In this case the elapsed time is measured using `CUDA.@elapsed`."

# ╔═╡ c6e66ff5-b5f2-4c0f-9eae-f123034bd438
begin
epochs_ = 15
with_terminal() do
	for epoch in 1:epochs_
	    # Training of one epoch
	    time = CUDA.@elapsed for (d, f) in gpu_train_loader
	        gs = gradient(() -> loss(f_model.(d, [gpu_model]), f), gpu_ps)
	        Flux.Optimise.update!(opt, gpu_ps, gs)
	    end
	    
	    # Report traning loss
	    println("Epoch: $(epoch), loss: $(global_loss(gpu_train_loader, gpu_model)), time: $(time)")
	end
end
end

# ╔═╡ 24f677cf-ee87-4af0-9c11-5d0911f1c700
md"### Test GPU results"

# ╔═╡ 04ff7b49-1feb-4b19-9b57-3ddad117427d
md"Again, the loss of the test data set is calculated using the root mean squared error (rmse)"

# ╔═╡ d300f5ce-2cce-4739-9108-ef20ff889955
@show "Test RMSE: $(global_loss(gpu_test_loader, gpu_model))"

# ╔═╡ ee8c4a30-b9e3-4987-b17c-69523037c749
md"The following plots show the forces computed by the neural network model and the surrogate model w.r.t. the norm of the position differences $|r_i-r_j|$. 30 forces are computed per each $|r_i-r_j|$."

# ╔═╡ f8a1b9de-46c9-4b9c-86ce-f71245fd020d
begin
	rss_gpu = []
	fs_model_gpu = []
	fs_surr_gpu = []
	for r in rs
		for _ in 1:30
			ϕ = rand() * 2.0 * π; θ = rand() * π
			x_ij = r * cos(ϕ) * sin(θ) 
			y_ij = r * sin(ϕ) * sin(θ) 
			z_ij = r * cos(θ)          
			r_ij = [x_ij, y_ij, z_ij]
			push!(rss_gpu, r)
			push!(fs_model_gpu, gpu_model(cu(r_ij)))
			push!(fs_surr_gpu, force(r_ij, lj))
		end
	end
end

# ╔═╡ a6999df8-850c-4ec7-8717-b4111cafebad
begin
	plot(rss_gpu, norm.(fs_model_gpu), seriestype = :scatter, markerstrokewidth=0,
		 xlabel = "Norm of the position difference, |ri-rj|", 
	  	 ylabel = "Norm of the force", 
		 label = "Neural network model force norm")
	plot!(rss_gpu, norm.(fs_surr_gpu), seriestype = :scatter, markerstrokewidth=0,
		  label = "Surrogate force norm")
end

# ╔═╡ 6ed86dde-63dd-429a-97f2-c901385f4bb7
begin
px_gpu =plot( [Array(f)[1] for f in fs_surr_gpu],
	          [Array(f)[1] for f in fs_model_gpu], 
			   seriestype = :scatter, markerstrokewidth=0, label="",
			   xlabel = "Fx, surrogate model", ylabel = "Fx, NN model")
py_gpu =plot( [Array(f)[2] for f in fs_surr_gpu],
	          [Array(f)[2] for f in fs_model_gpu], 
			   seriestype = :scatter, markerstrokewidth=0, label="",
			   xlabel = "Fy, surrogate model", ylabel = "Fy, NN model")
pz_gpu =plot( [Array(f)[3] for f in fs_surr_gpu],
	          [Array(f)[3] for f in fs_model_gpu], 
			   seriestype = :scatter, markerstrokewidth=0, label="",
			   xlabel = "Fz, surrogate model", ylabel = "Fz, NN model")
plot(px_gpu, py_gpu, pz_gpu)
end

# ╔═╡ Cell order:
# ╟─5dd8b98b-967c-46aa-9932-25157a10d0c2
# ╟─09985247-c963-4652-9715-1f437a07ef81
# ╟─3bdb681d-f9dc-4d37-8667-83fccc247b3d
# ╟─e0e8d440-1df0-4581-8277-d3d6886351d7
# ╟─29e753a9-f49d-41b4-85f3-e8f35d2d67f5
# ╠═a904c8dc-c7fa-4ef7-97f0-9018bb9f2db2
# ╟─6a5f238c-0672-4fcb-9da9-3c2f5a718d5a
# ╠═5f6d4c16-bf9f-442a-bb43-5cbe1b861fb1
# ╟─b562c28b-a530-41f8-b2d4-57e2df2860c5
# ╟─7bcb0360-a656-4637-b0ce-e4ada1e9ce0a
# ╠═3e606585-86b6-4389-818d-bcbdb6078608
# ╟─19dc1efa-27ac-4c94-8f4d-bc7886c2959e
# ╟─c0d477c3-3af8-4b19-bf84-657d4c60fea8
# ╟─4430121b-9d15-4c12-86f8-48fa1579845b
# ╟─160f3fb3-8dbd-4a3b-befe-1bef361fdd69
# ╠═67d9544a-4ff4-4703-8579-b8a335d58f63
# ╟─0fa4ca5a-f732-4630-991c-ff5e4b76c536
# ╟─40c9d9cd-05af-4bbf-a772-5c09c1b03a66
# ╟─93450770-74c0-42f7-baca-7f8276373f9f
# ╟─216f84d3-ec5c-4c49-a9d7-af0e98907e15
# ╠═6e20dcb0-2fed-4637-ae07-775f3cd4a8dd
# ╟─8aa8f56a-24a1-4ddd-840e-91517fd27b9c
# ╠═8787b0dc-81f5-4e15-8931-4f8325d45061
# ╟─152f05fb-ac0a-4d8d-8975-94585443853a
# ╟─fe7cf55f-cbc9-41db-8554-03fdecbc881d
# ╟─dd5e6c1f-ad9b-49fa-8b5e-9cffd5781c16
# ╠═7b48d757-18eb-4f6b-bf7f-53d8c52cf7a1
# ╟─f93f298d-6ac0-45e2-abee-ab36d85b3f7b
# ╟─1180e03c-482e-428a-b66c-5534bcb4de83
# ╟─08fbcf33-0935-41ce-a721-c597e2570b47
# ╠═8a03ffbc-9ab2-438d-b71b-1d7238b33507
# ╟─2c629ed0-e21d-4f38-b46f-fc2a1757b62f
# ╟─48e7fa16-ea5f-44f5-84ab-2994e7382c4e
# ╠═7410e309-2ee0-4b65-af2a-c4f738a3bd80
# ╟─2244cd54-f927-40e0-8fad-75a34096044d
# ╟─b481e2f6-1a87-4af3-8d93-111e2ab8d933
# ╟─01de26b8-c596-4690-bb62-07170260ddbc
# ╠═93e7f0c4-ad8c-473e-a8d3-09508f2ba6df
# ╟─f1a29169-2c8a-41bc-9c35-be2c77b418ec
# ╠═54891249-99be-4798-a0e9-5b0193ecaa3c
# ╟─1cb8b6d0-d645-4d16-a501-cc9d2a047b39
# ╠═c8c43e19-daff-4272-8703-a2dcaceca7a9
# ╟─fcccc8ca-97c8-43cd-ab1d-b76f126ab617
# ╠═49d7f00a-db88-4dab-9209-c9840255d999
# ╟─4481fcae-6c0c-4455-9a11-0eb1f19805cc
# ╠═2e199532-5177-4b39-9c7e-83d61c1e2f13
# ╟─74d94c45-7a66-4779-9bd8-d6987d23bdc4
# ╟─46a7f1ab-5f99-4b69-9bff-c299815cccab
# ╠═195925f3-19ea-47a2-bc31-561bf698f580
# ╟─97cd889b-7fd5-494a-8f34-5830e53557f6
# ╠═abf7c48c-221b-43d7-a583-716b192bc5f6
# ╠═276e1787-d15e-4094-bf00-30b6192228e1
# ╠═d22ab6ee-4184-4dfc-a9bb-2af4f0de5882
# ╟─d720916f-8ed8-4caf-bcf1-43da2f78e21a
# ╟─4d94701b-e3a5-4aef-ba23-97c1150f89a1
# ╟─c2492af2-23e7-45c5-bf44-109c59d7986d
# ╠═d4526269-07e6-4684-8bed-7cee5898ef52
# ╟─a67f8826-ee85-4e74-aec5-dd2c79c56c13
# ╠═5a7e8a3f-3d8d-4048-80fe-c64e9c1cfd44
# ╟─f89622bb-433a-4547-aec0-9fd3a2ffbaf2
# ╠═c6e66ff5-b5f2-4c0f-9eae-f123034bd438
# ╟─24f677cf-ee87-4af0-9c11-5d0911f1c700
# ╟─04ff7b49-1feb-4b19-9b57-3ddad117427d
# ╠═d300f5ce-2cce-4739-9108-ef20ff889955
# ╟─ee8c4a30-b9e3-4987-b17c-69523037c749
# ╠═f8a1b9de-46c9-4b9c-86ce-f71245fd020d
# ╠═a6999df8-850c-4ec7-8717-b4111cafebad
# ╠═6ed86dde-63dd-429a-97f2-c901385f4bb7
