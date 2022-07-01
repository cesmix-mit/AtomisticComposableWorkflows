macro savevar(path, var)
    quote
        write("$(path)" * $(string(var)) * ".dat", string($(esc(var))))
    end
end
