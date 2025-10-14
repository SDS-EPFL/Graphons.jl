module SuiteSparseGraphBLASExt

using Random
import SuiteSparseGraphBLAS: GBMatrix
import Graphons: rand, _rand!, AbstractGraphon, make_empty_graph, clear_graph!

function make_empty_graph(::Type{GB}, n) where {GB<:GBMatrix}
    return GB(n, n)
end

function clear_graph!(A::GBMatrix)
    empty!(A)
end

end
