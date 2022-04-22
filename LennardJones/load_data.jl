# Save β
using ACE1
using JuLIP
function save_β_to_file(β, rpi_params, directory_name)
        try
            mkdir("ACE_MD/$(directory_name)/")
        catch
        
    end
    basis = get_rpi(rpi_params)
    IP = JuLIP.MLIPs.combine(basis, β)
    ACE1.Export.export_ace("ACE_MD/$(directory_name)/parameters.ace", IP)

    # Make adjustment for package name ACE1.jl -> ACE.jl
    fff = readlines("ACE_MD/$(directory_name)/parameters.ace")
    fff[10] = "radbasename=ACE.jl.Basic"
    open("ACE_MD/$(directory_name)/parameters.ace", "w") do io
        for l in fff
            write(io, l*"\n")
        end
    end
end

# Load data
function load_data(file; max_entries = 2000 )
    systems  = AbstractSystem[]
    energies = Float64[]
    forces    = Vector{SVector{3, Float64}}[]
    stresses = SVector{6, Float64}[]
    open(file, "r") do io
        count = 1
        while !eof(io) && (count <= max_entries)
            # Read info line
            line = readline(io)
            num_atoms = parse(Int, line)
            
            line = readline(io)
            lattice_line = match(r"Lattice=\"(.*?)\" ", line).captures[1]
            lattice = parse.(Float64, split(lattice_line)) * 1u"Å"
            box = [lattice[1:3],
                        lattice[4:6], 
                        lattice[7:9]]
            bias = -5.0 
            try 
                energy_line = match(r"energy=(.*?) ", line).captures[1]
                energy = parse(Float64, energy_line)
                push!(energies, energy)
            catch
                push!(energies, NaN)
            end
            
            try
                stress_line = match(r"stress=\"(.*?)\" ", line).captures[1]
                stress = parse.(Float64, split(stress_line))
                push!(stresses, SVector{6}(stress))
            catch
                push!(stresses, SVector{6}( fill(NaN, (6,))))
            end

            bc = []
            try
                bc_line = match(r"pbc=\"(.*?)\"", line).captures[1]
                bc = [ t=="T" ? Periodic() : DirichletZero() for t in split(bc_line) ]
            catch
                bc = [DirichletZero(), DirichletZero(), DirichletZero()]
            end

            properties = match(r"Properties=(.*?) ", line).captures[1]
            properties = split(properties, ":")
            properties = [properties[i:i+2] for i = 1:3:(length(properties)-1)]
            atoms = Vector{Atom}(undef, num_atoms)
            force = Vector{SVector{3, Float64}}(undef, num_atoms) 
            for i = 1:num_atoms
                line = split(readline(io))
                line_count = 1
                position =  0.0
                element = 0.0
                data = Dict( () )
                for prop in properties
                    if prop[1] == "species"
                        element = Symbol(line[line_count])
                        line_count += 1 
                    elseif prop[1] == "pos"
                        position = SVector{3}(parse.(Float64, line[line_count:line_count+2])) .- bias
                        line_count += 3
                    elseif prop[1] == "forces"
                        force[i] = SVector{3}(parse.(Float64, line[line_count:line_count+2]))
                        line_count += 3
                    else
                        length = parse(Int, prop[3])
                        if length == 1
                            data = merge(data, Dict( (prop[1] => line[line_count]) ) )
                        else
                            data = merge(data, Dict( (prop[1] => line[line_count:line_count+length-1]) ) )
                        end
                    end
                end
                atoms[i] = Atom(element, position * 1u"Å", data = data) 
                 
            end

            push!(forces, force)
            system = FlexibleSystem(atoms, box, bc)
            push!(systems, system)
            count += 1
        end
        println(count)
    end
    return systems, energies, forces, stresses
end



