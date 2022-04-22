function train_test_split(total_set; train_ratio = 0.7)
    @assert (train_ratio < 1.0) & (train_ratio > 0.0) 
    n_total = length(first(total_set))
    n_train = floor(Int, train_ratio*n_total)
    n_test  = floor(Int, (1-train_ratio)*n_total)
    rand_list = randperm(n_total)
    train_index, test_index = rand_list[1:n_train], rand_list[n_train+1:n_train+n_test]

    training_set = []
    testing_set = []
    for item in total_set
        push!(training_set, item[train_index])
        push!(testing_set, item[test_index])
    end
    return training_set, testing_set
end

function distance_histogram(system::AbstractSystem; r = LinRange(0.5, 5.0, 200)*u"Å")
    d = Float64[]
    positions = position(system)
    n = length(positions)
    for i = 1:n
        ri = positions[i]
        for j = (i+1):n
            append!(d, ustrip(norm(ri - positions[j])))
        end
    end
    d
end

function distance_histogram(systems::Vector{<:AbstractSystem}; r = LinRange(0.5, 5.0, 200)*u"Å")
    d = Float64[]
    for sys in systems 
        append!(d, distance_histogram(sys; r))
    end
    return d
end


function get_data(set, basis)
    if length(set) == 4
        systems, energies, forces, _ = set
    elseif length(set) == 3
        systems, energies, forces = set
    end
    num_systems = length(systems)

    num_atoms = length(first(systems))
    e = zeros(num_systems)
    f = zeros(num_systems*num_atoms*3)

    B = zeros(num_systems, length(basis))
    dB = zeros(num_systems*num_atoms*3, length(basis))

    for i = 1:num_systems
        sys = systems[i]
        BB = reshape(evaluate_basis(sys, basis), 1, :)
        B[i, :] = BB

        dBB = vcat(evaluate_basis_d(sys, basis)...)
        dB[(i-1)*3*num_atoms+1:i*3*num_atoms, :] = dBB
    
        e[i] = energies[i]
        f[(i-1)*3*num_atoms+1:i*3*num_atoms] = [fi for fi in vcat(vcat(forces[i]...)...)]
    end
    e, f, B, dB
end


## Calculate descriptors 
function get_data_random_batch(set, batch_size, basis)
    if length(set) == 4
        systems, energies, forces, _ = set
    elseif length(set) == 3
        systems, energies, forces = set
    end
    num_systems = length(systems)
    rand_list = randperm(num_systems)
    indices = rand_list[1:min(num_systems, batch_size)]

    num_atoms = length(first(systems))
    e = zeros(batch_size)
    f = zeros(batch_size*num_atoms*3)

    B = zeros(batch_size, length(basis))
    dB = zeros(batch_size*num_atoms*3, length(basis))
    for i = 1:batch_size
        sys = systems[indices[i]]
        BB = reshape(evaluate_basis(sys, basis), 1, :)
        B[i, :] = BB

        dBB = vcat(evaluate_basis_d(sys, basis)...)
        dB[(i-1)*3*num_atoms+1:i*3*num_atoms, :] = dBB
    
        e[i] = energies[indices[i]]
        f[(i-1)*3*num_atoms+1:i*3*num_atoms] = [fi for fi in vcat(vcat(forces[indices[i]]...)...)]
    end
    e, f, B, dB
end

# Calculate coefficients
function estimate_β(A, b, Qinv)
    K = Symmetric(A'*Qinv*A)
    y = A'*Qinv*b
    K \ y
end

function joint_neg_log_likelihood(x, p)
    A, e, f = p
    β = x[3:end]
    d = [exp(x[1]).+ 0.0 * e; exp(x[2]) .+ 0.0*f]
    Q = Diagonal(d)
    p = MvNormal(A*β, Q)

    -logpdf(p, [e; f])
end

function joint_neg_log_likelihood_dpp(x, p)
    dpp, batch_size = p
    e, f, B, dB  = get_data_dpp_batch(dpp, batch_size)
    A = [B; dB]
    β = x[3:end]
    d = [exp(x[1]).+ 0.0 * e; exp(x[2]) .+ 0.0*f]
    Q = Diagonal(d)
    p = MvNormal(A*β, Q)
    -logpdf(p, [e; f])
end

function neg_log_likelihood(x, p)
    A, e, f, Q = p
    β = x
    p = MvNormal(A*β, Q)
    -logpdf(p, [e; f])
end

function var_neg_log_likelihood(x, p)
    A, e, f = p
    d = [exp(x[1]).+ 0.0 * e; exp(x[2]) .+ 0.0*f]
    Q = Diagonal(d)
    b = [e;f]
    β = inv(A' * (Q\A)) * (A' * (Q\b)) 
    p = MvNormal(A*β, Q)
    -logpdf(p, b)
end

function var_neg_test_likelihood(x, p)
    Atr, etr, ftr, Ate, ete, fte  = p
    d = [exp(x[1]).+ 0.0 * etr; exp(x[2]) .+ 0.0*ftr]
    Q = Diagonal(d)
    β = (Atr' * (Q\Atr)) \ (Atr' * (Q\[etr;ftr])) 

    d = [exp(x[1]).+ 0.0 * ete; exp(x[2]) .+ 0.0*fte]
    Q = Diagonal(d)
    p = MvNormal(Ate*β, Q)
    err = Ate*β - [ete;fte]
    err' * (Q \ err) + logdet(Q) / length(d)
end

struct DPPKernel{T} 
    L :: EllEnsemble{T}
    e :: Vector{T}
    f :: Vector{T}
    B :: Matrix{T}
    dB :: Matrix{T}
end

function DPPKernel(training_data, rpi_params)
    e, f, B, dB = get_data(training_data, rpi_params)

    n = size(B, 1)
    num_atoms = (size(dB, 1) ÷n ) ÷ 3
    K = zeros(n, n)
    for i = 1:n
        for j = i:n
            a = B[i, :] / norm(B[i, :])
            b = B[j, :] / norm(B[j, :])
            K[i, j] = (a'*b)^2

            a = vec(dB[ (3*num_atoms)*(i-1)+1:(3*num_atoms)*(i), : ])
            a = a / norm(a)
            b = vec(dB[ (3*num_atoms)*(j-1)+1:(3*num_atoms)*(j), : ])
            b = b / norm(b)
            K[i, j] *= (a'*b)^2
        end
    end
    K += 1e-8 * I(n)
    K = Symmetric(K)

    ## Alternate kernel 
    # A = [B; dB]
    # C = inv(A'*A) * (2*n-1)
    # Bvec = [ B[i, :] for i = 1:n]
    # kernel(x, y) = exp(-0.5*(x-y)'*(C*(x-y)) )
    # K = Symmetric(hcat( [[kernel(x, y) for x in Bvec] for y in Bvec]... ))

    DPPKernel(EllEnsemble(K), e, f, B, dB)
end


function get_data_dpp_indices(dpp::DPPKernel, indices :: Vector{Int})
    e = dpp.e[indices]
    f = reduce(vcat, [dpp.f[i:i+39-1] for i in indices])

    B = dpp.B[indices, :]
    dB = reduce(vcat, [dpp.dB[i:i+39-1, :] for i in indices])
    e, f, B, dB 
end

function get_data_dpp_batch(dpp :: DPPKernel, batch_size :: Int)
    indices = DPP.sample(dpp.L, batch_size)
    get_data_dpp_indices(dpp, indices)
end

function get_data_dpp_mode(dpp :: DPPKernel, batch_size :: Int)
    resize!(dpp.L, batch_size)
    indices = greedy_subset(dpp.L, batch_size)
    get_data_dpp_indices(dpp, indices)
end



