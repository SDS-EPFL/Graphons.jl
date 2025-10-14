using Graphons
using Documenter

using Literate

LITERATE_INPUT = joinpath(@__DIR__, "literate")
LITERATE_OUTPUT = joinpath(@__DIR__, "src")

for dir_path in filter(isdir, readdir(joinpath(@__DIR__, "literate"), join=true))
    dirname = basename(dir_path)

    for (root, _, files) in walkdir(dir_path), file in files
        # ignore non julia files
        splitext(file)[2] == ".jl" || continue
        # full path to a literate script
        ipath = joinpath(root, file)
        # generated output path
        opath = splitdir(replace(ipath, LITERATE_INPUT => LITERATE_OUTPUT))[1]
        # generate the markdown file calling Literate
        Literate.markdown(ipath, opath)
    end
end

DocMeta.setdocmeta!(Graphons, :DocTestSetup, :(using Graphons); recursive=true)

makedocs(;
    modules=[Graphons],
    authors="Charles Dufour,Jake Grainger",
    repo="https://github.com/SDS-EPFL/Graphons.jl/blob/{commit}{path}#{line}",
    sitename="Graphons.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://SDS-EPFL.github.io/Graphons.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => ["First steps" => "tutorials/01_simple_graphon.md",
            "Multiplex networks" => "tutorials/02_multiplex_networks.md"],
        "Design" => ["Graph representation" => "graphs_type.md"],
        "Index" => "internal.md",
    ],
)

deploydocs(;
    repo="github.com/SDS-EPFL/Graphons.jl",
    devbranch="main",
)
