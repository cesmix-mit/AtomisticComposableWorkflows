macro savevar(path, var)
    quote
        write("$(path)" * $(string(var)) * ".dat", string($(esc(var))))
    end
end

macro savefig(path, var)
    return :( savefig($(var), "$(path)" * $(string(var)) * ".png") )
end

