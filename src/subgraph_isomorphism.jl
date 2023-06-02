"""
    subgraph_density(graphon, subgraph_edge_list; method = CubaCuhre(), kwargs...)

General non-optimized subgraph isomorphism algorithm.

```math
    t_{\\text {hom }}(F, w)=\\int_{[0,1]^k} \\prod\\limits_{(i, j) \\in E(F)}
    w\\left(u_i, u_j\\right) du_{1: k}
```

Compute the density of a subgraph ``F`` in a graphon ``w`` by integrating over the graphon.
This does not try to optimize the integration (e.g. factorization, block model, subgraph
structure, etc.)

Works for any subgraph, but is slow for large subgraphs.
"""
function subgraph_density(graphon::AbstractGraphon,
                          subgraph_edge_list::Vector{Tuple{Int, Int}}; method = CubaCuhre(),
                          kwargs...)
    f(u, p) = prod([_probs(graphon, u[i], u[j]) for (i, j) in subgraph_edge_list])
    prob = IntegralProblem(f, zeros(length(subgraph_edge_list)),
                           ones(length(subgraph_edge_list)))
    return solve(prob, method; kwargs...).u
end

function cycle_density(graphon, k = 3; kwargs...)
    subgraph_density(graphon, [(i, i % k + 1) for i in 1:k]; kwargs...)
end
