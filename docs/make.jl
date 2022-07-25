pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add AtomisticComposableWorkflows to environment stack

using Documenter
using DocumenterCitations
using Literate

makedocs(
    authors = "CESMIX-MIT",
    repo = "https://github.com/cesmix-mit/AtomisticComposableWorkflows/blob/{commit}{path}#{line}",
    sitename = "AtomisticComposableWorkflows",
    pages = [
        "Home" => "index.md",
    ],
    doctest = true,
    linkcheck = true,
    strict = false
)

deploydocs(;
    repo = "github.com/cesmix-mit/AtomisticComposableWorkflows",
    devbranch = "main",
    push_preview = true
)
