# Graph Types

When creating `Graphon` objects, we can specify the type of graph to be
sampled. By default, the sampled graphs are dense matrices. However, we can
specify `SparseMatrixCSC` to obtain sparse graphs (should be supported by
default), but other types are also possible.

!!! info "Work in Progress"

    This section is a work in progress. More graph types will be added in the future.

## Design Philosophy

One of the main difficulties is that we cannot know what value is used to
represent a non-edge in a decorated graph type. For example, for a simple graph
encoded in a `BitMatrix`, a non-edge is represented by `false`. But for a
weighted graph encoded in a `Matrix{Float64}`, there is no necessarily a
default value for a non-edge. It could be `0.0`, `NaN`, or any other value.

To circumvent this issue, we have decided to let the user define how to create
an empty graph of a given type if the default is not suitable. This means that
the function to randomly sample a graph with preallocated memory now expects
that the input graph is empty and of the right size.

```@docs
Graphons._rand!
```

## Custom Graph Types

To sample with your own graph type, you need to redefine the function
`make_empty_graph(::Type{M}, n)`, such that it returns an empty graph of type
`M` with `n` nodes and no edges.

!!! info "Design Choice"

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
f_sparseBLAS = SimpleContinuousGraphon((x,y)-> 0.1, GBMatrix{Bool})
A = rand(f_sparseBLAS, 41)
```
