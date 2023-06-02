using Graphon
using Documenter

DocMeta.setdocmeta!(Graphon, :DocTestSetup, :(using Graphon); recursive = true)

makedocs(;
         modules = [Graphon],
         authors = "Jake Grainger, Charles Dufour",
         repo = "https://github.com/JakeGrainger/Graphon.jl/blob/{commit}{path}#{line}",
         sitename = "Graphon.jl",
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://JakeGrainger.github.io/Graphon.jl",
                                  edit_link = "main",
                                  assets = String[]),
         pages = [
             "Home" => "index.md",
         ])

deploydocs(;
           repo = "github.com/JakeGrainger/Graphon.jl",
           devbranch = "main")
