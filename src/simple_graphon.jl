## Continuous graphons

"""
    SimpleContinuousGraphon{M,F} <: SimpleGraphon{M}

A continuous graphon represented by a function `f(x, y)` that returns the probability
of an edge between nodes at latent positions `x` and `y` in [0,1].

# Type Parameters
- `M`: The matrix type used to represent sampled graphs (e.g., `BitMatrix`, `SparseMatrixCSC{Bool,Int}`)
- `F`: The type of the graphon function

# Fields
- `f::F`: A callable function `f(x, y) -> Float64` where x, y ∈ [0,1] and the return value is in [0,1]

# Constructors
    SimpleContinuousGraphon(f, M=BitMatrix)

Create a continuous graphon from function `f` with matrix type `M`.

# Examples
```julia
# Constant probability graphon
g = SimpleContinuousGraphon((x, y) -> 0.5)

# Distance-based graphon
g = SimpleContinuousGraphon((x, y) -> exp(-abs(x - y)))

# Sparse matrix representation
g = SimpleContinuousGraphon((x, y) -> 0.1, SparseMatrixCSC{Bool,Int})
```

# See Also
- [`SBM`](@ref): Discrete stochastic block model
- [`empirical_graphon`](@ref): Discretize a continuous graphon
"""
struct SimpleContinuousGraphon{M,F} <: SimpleGraphon{M}
    f::F
end

(g::SimpleContinuousGraphon)(x, y) = g.f(x, y)

function SimpleContinuousGraphon(f::F, M=BitMatrix) where {F}
    return SimpleContinuousGraphon{M,F}(f)
end


## Stochastic Block Models

"""
    SBM{P,S,S2} <: AbstractGraphon{Bool,BitMatrix}

A Stochastic Block Model (SBM) is a discrete graphon with `K` blocks where edge
probabilities depend only on the block membership of nodes.

# Fields
- `θ::P`: K×K matrix of edge probabilities between blocks, where θ[i,j] is the probability of an edge between block i and j
- `size::S`: Vector of block sizes (must sum to 1), representing the proportion of nodes in each block
- `cumsize::S2`: Cumulative sum of block sizes for efficient block assignment

# Constructor
    SBM(θ, sizes)

Create a stochastic block model with edge probability matrix `θ` and block sizes `sizes`.

# Arguments
- `θ`: K×K matrix where each entry is in [0,1]
- `sizes`: Vector of K positive values that sum to 1

# Examples
```julia
# Two-block assortative model
θ = [0.8 0.1; 0.1 0.8]
sizes = [0.5, 0.5]
sbm = SBM(θ, sizes)

# Three-block model with unequal sizes
θ = [0.9 0.1 0.2; 0.1 0.8 0.1; 0.2 0.1 0.7]
sizes = [0.3, 0.5, 0.2]
sbm = SBM(θ, sizes)
```

# See Also
- [`SimpleContinuousGraphon`](@ref): Continuous graphon representation
- [`empirical_graphon`](@ref): Convert continuous graphon to SBM
"""
struct SBM{P,S,S2} <: AbstractGraphon{Bool,BitMatrix}
    θ::P
    size::S
    cumsize::S2
end

function SBM(θ, sizes)
    @argcheck all(x -> x <= 1 && x >= 0, θ)
    @argcheck all(sizes .> 0)
    cumsizes = cumsum(sizes)
    @argcheck last(cumsizes) ≈ 1
    return SBM(θ, sizes, cumsizes)
end

function (g::SBM)(x, y)
    latents_x = _convert_latent_to_block(g, x)
    latents_y = _convert_latent_to_block(g, y)
    return g.θ[latents_x, latents_y]
end


"""
    empirical_graphon(f::SimpleContinuousGraphon, k::Int) -> SBM

Discretize a continuous graphon into a Stochastic Block Model with `k` blocks
by evaluating the graphon at a uniform grid of points.

# Arguments
- `f`: A continuous graphon
- `k`: Number of blocks for discretization

# Returns
An [`SBM`](@ref) with `k` blocks approximating the continuous graphon.

# Examples
```julia
# Create continuous graphon
f = SimpleContinuousGraphon((x, y) -> min(x, y))

# Discretize into 10-block SBM
sbm = empirical_graphon(f, 10)
```

# See Also
- [`SimpleContinuousGraphon`](@ref): Continuous graphon type
- [`SBM`](@ref): Stochastic block model type
"""
function empirical_graphon(f::SimpleContinuousGraphon, k::Int)
    ξs = range(0, stop=1, length=k)
    sizes = fill(1 / k, k)
    # dirty hack to ensure sum(sizes) == 1
    sizes[end] += 1 - sum(sizes)
    θ = [f(ξs[i], ξs[j]) for i in 1:k, j in 1:k]
    return SBM(θ, sizes)
end
