# Graph Types

When creating `Graphon` objects, we can specify the type of graph to be
sampled. By default, the sampled graphs are dense matrices. However, we can
specify `SparseMatrixCSC` to obtain sparse graphs (should be supported by
default), but other types are also possible.

!!! info "Work in Progress"

    This section is a work in progress. More graph types will be added in the future.

To sample with your own graph type, you need to redefine the function
`make_empty_graph(::Type{M}, n)`, such that it returns an empty graph of type
`M` with `n` nodes and no edges.

!!! warning "Design Choice"

    Since for decorated graphs it is sometimes hard to know in advance what will be the default
    representation of a non-edge, we have left that choice to the user via the `make_empty_graph`
    function.

An example of a custom graph type is provided in `ext/SuiteSparseGraphBLAS.jl`,
which uses the `SuiteSparseGraphBLAS` package to create large sparse graphs
efficiently.

```julia
function make_empty_graph(::Type{GB}, n) where {GB<:GBMatrix}
    return GB(n, n)
end
```

This allows us to create a `Graphon` object that will sample graphs of type
`GBMatrix`:

```julia
using SuiteSparseGraphBLAS
M2 = GBMatrix{Bool,Bool}
f_sparseBLAS = SimpleContinuousGraphon((x,y)-> 0.1, M2)
rand(f_sparseBLAS, 41)
```
