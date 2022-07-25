    pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add AtomisticComposableWorkflows to environment stack

using AtomisticComposableWorkflows
using Documenter
using DocumenterCitations
using Literate

DocMeta.setdocmeta!(AtomisticComposableWorkflows, :DocTestSetup, :(using AtomisticComposableWorkflows); recursive = true)

bib = CitationBibliography(joinpath(@__DIR__, "citations.bib"))

# Generate examples

#const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
#const OUTPUT_DIR = joinpath(@__DIR__, "src/generated")

#examples = Pair{String,String}[]

#for (_, name) in examples
#    example_filepath = joinpath(EXAMPLES_DIR, string(name, ""))
#    Literate.markdown(example_filepath, OUTPUT_DIR, documenter = true)
#end

#examples = [title => joinpath("generated", string(name, ".md")) for (title, name) in examples]

makedocs(bib;
    modules = [AtomisticComposableWorkflows],
    authors = "CESMIX-MIT",
    repo = "https://github.com/cesmix-mit/AtomisticComposableWorkflows/blob/{commit}{path}#{line}",
    sitename = "AtomisticComposableWorkflows",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://cesmix-mit.github.io/AtomisticComposableWorkflows",
        assets = String[],
    ),
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
