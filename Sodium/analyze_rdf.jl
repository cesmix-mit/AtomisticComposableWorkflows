function read_rdf(file::String)
    lines = readlines(file)[4:end]
    number_of_lines = length(lines)
    number_of_bins = parse(Int, split(lines[1])[2])
    number_of_rdfs = number_of_lines ÷ (number_of_bins+1)

    bins = Float64[]
    rdf = Vector{Float64}[]
    count = 2
    for j = 1:number_of_rdfs
        rdf_ = Float64[]
        for k = 1:number_of_bins
            r = split(lines[count])
            if j == 1
                push!(bins, parse(Float64, r[2]))
            end
            push!(rdf_, parse(Float64, r[3]))
            count +=1
        end
        count +=1
        push!(rdf, rdf_)
    end

    return bins, rdf
end

bins, rdf_warm = read_rdf("TEMP/tmp_warm.rdf")
bins, rdf_melt = read_rdf("TEMP/tmp_melt.rdf")
bins, rdf_heat = read_rdf("TEMP/tmp_heat.rdf")
bins, rdf_hot = read_rdf("TEMP/tmp_hot.rdf")



using GLMakie

f = Figure()
ax = Axis(f[1,1], xlabel = "r", ylabel = "g(r)", title = "RDF of Sodium Clusters")
for j = 1:10
    lines!(ax, bins, rdf_warm[j], color = :blue, label = "100K")
    lines!(ax, bins, rdf_melt[j], color = :lightblue, label = "300K")
    lines!(ax, bins, rdf_heat[j], color = :orange, label = "500k")
    lines!(ax, bins, rdf_hot[j], color = :red, label = "1000k")
end
axislegend(ax, merge = true)
