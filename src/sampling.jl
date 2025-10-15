## random unobserved latents

rand(f::AbstractGraphon, n::Int) = rand(Random.default_rng(), f, n)


"""
    rand([rng::AbstractRNG], graphon::AbstractGraphon{T,M}, n::Int) -> M

Generate a random graph from a graphon with `n` nodes.

Latent positions for each node are drawn uniformly at random from [0,1], and edges
are sampled according to the graphon. For simple graphons, edges are present with
probability `f(ξᵢ, ξⱼ)`. For decorated graphons, edge weights are sampled from
the distribution `f(ξᵢ, ξⱼ)`.

# Arguments
- `rng`: Random number generator (optional, defaults to `Random.default_rng()`)
- `graphon`: A graphon model (e.g., [`SimpleContinuousGraphon`](@ref), [`SBM`](@ref), [`DecoratedGraphon`](@ref))
- `n`: Number of nodes in the graph

# Returns
An adjacency matrix of type `M` (determined by the graphon) with:
- Symmetric structure: `A[i,j] == A[j,i]`
- No self-loops: `A[i,i] == 0` for all i
- Edge type `T` (Bool for simple graphs, Float64 for weighted, etc.)

# Examples
```julia
# Simple random graph
g = SimpleContinuousGraphon((x, y) -> 0.3)
A = rand(g, 100)  # 100-node graph with 30% edge probability

# Reproducible sampling
using Random
rng = MersenneTwister(42)
A = rand(rng, g, 100)

# Stochastic block model
sbm = SBM([0.8 0.1; 0.1 0.8], [0.5, 0.5])
A = rand(sbm, 200)

# Decorated graphon with weighted edges
using Distributions
dg = DecoratedGraphon((x, y) -> Normal(x + y, 0.1))
W = rand(dg, 50)  # Returns Matrix{Float64}
```

# See Also
- [`sample_graph`](@ref): Sample with fixed latent positions
- [`SimpleContinuousGraphon`](@ref), [`SBM`](@ref), [`DecoratedGraphon`](@ref)
"""
function rand(rng::AbstractRNG, f::AbstractGraphon{T,M}, n::Int) where {T,M}
    return _rand!(rng, f, make_empty_graph(M, n), Base.rand(rng, n))
end

"""
    _rand!(rng::AbstractRNG, f::AbstractGraphon{T,M}, A::M, ξs)

    Generates a random graph according to the graphon `f` and the latent positions `ξs`.
    The generated graph is stored in `A`.

!!! warning
    This function expects that `A` is an empty graph of the right size and type. It does not
    try to clean it up before filling it. See `make_empty_graph` for more details.
"""
function _rand!(rng::AbstractRNG, f::AbstractGraphon{T,M}, A::M, ξs) where {T,M}
    for j in axes(A, 2)
        for i in axes(A, 1)
            if i < j
                A[i, j] = rand(rng, f(ξs[i], ξs[j]))
                A[j, i] = A[i, j]
            end
        end
    end
    return A
end

function _rand!(rng::AbstractRNG, f::SimpleGraphon{M}, A::M, ξs) where {M}
    for j in axes(A, 2)
        for i in axes(A, 1)
            if i < j && Base.rand(rng) < f(ξs[i], ξs[j])
                A[i, j] = one(eltype(A))
                A[j, i] = A[i, j]
            end
        end
    end
    return A
end



## known latents

sample_graph(f::AbstractGraphon, args...) = sample_graph(Random.default_rng(), f, args...)


"""
    sample_graph([rng::AbstractRNG], graphon::AbstractGraphon, n::Int) -> M
    sample_graph([rng::AbstractRNG], graphon::AbstractGraphon, ξs::AbstractVector) -> M

Generate a graph from a graphon with deterministic or specified latent positions.

This function provides more control than [`rand`](@ref) by allowing you to:
1. Use evenly-spaced latent positions `ξ = [0, 1/(n-1), 2/(n-1), ..., 1]`
2. Specify custom latent positions for each node

Graphs sampled with the same latent positions will have the same expected structure,
making this useful for reproducibility and controlled experiments.

# Arguments
- `rng`: Random number generator (optional)
- `graphon`: A graphon model
- `n`: Number of nodes (latents will be evenly spaced on [0,1])
- `ξs`: Vector of latent positions in [0,1], one per node

# Returns
An adjacency matrix of type `M` with symmetric structure and no self-loops.

# Examples
```julia
# Evenly-spaced latents
g = SimpleContinuousGraphon((x, y) -> x * y)
A = sample_graph(g, 10)  # Uses ξ = [0, 0.111..., 0.222..., ..., 1]

# Custom latent positions
ξs = [0.1, 0.2, 0.5, 0.9]
A = sample_graph(g, ξs)

# Reproducible with same latents
using Random
rng = MersenneTwister(42)
A1 = sample_graph(rng, g, ξs)
rng = MersenneTwister(42)
A2 = sample_graph(rng, g, ξs)
# A1 and A2 will be identical

# Compare random vs deterministic latents
A_random = rand(g, 100)        # Random latents
A_fixed = sample_graph(g, 100) # Evenly-spaced latents
```

# See Also
- [`rand`](@ref): Sample with random latent positions
- [`SimpleContinuousGraphon`](@ref), [`SBM`](@ref), [`DecoratedGraphon`](@ref)
"""
function sample_graph(rng::AbstractRNG, f::AbstractGraphon, n::Int)
    return sample_graph(rng, f, range(0, 1, length=n))
end

function sample_graph(rng::AbstractRNG, f::AbstractGraphon{T,M}, ξs) where {T,M}
    n = length(ξs)
    return _rand!(rng, f, make_empty_graph(M, n), ξs)
end
