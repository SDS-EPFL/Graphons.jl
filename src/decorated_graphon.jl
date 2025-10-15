_infer_eltype(d) = eltype(d)

function _infer_eltype(d::MultivariateDistribution)
    return SizedVector{length(d),eltype(d)}
end


## Continuous decorated graphons

"""
    DecoratedGraphon{T,M,F,D} <: AbstractGraphon{T,M}

A decorated graphon where edges have rich attributes drawn from distributions.
Instead of returning edge probabilities, the graphon function returns a distribution
from which edge values are sampled.

# Type Parameters
- `T`: The type of edge values (inferred from the distribution)
- `M`: The matrix type for sampled graphs
- `F`: The type of the graphon function
- `D`: The type of distributions returned by the function

# Fields
- `f::F`: A callable function `f(x, y) -> Distribution` where x, y ∈ [0,1]

# Constructors
    DecoratedGraphon(f)
    DecoratedGraphon(f, M)

Create a decorated graphon from function `f`. The edge type is automatically inferred
by evaluating `f(0.1, 0.2)`.

# Examples
```julia
using Distributions

# Edges are normal random variables
g = DecoratedGraphon((x, y) -> Normal(x + y, 0.1))

# Edges are Poisson counts
g = DecoratedGraphon((x, y) -> Poisson(10 * x * y))

# With custom matrix type
g = DecoratedGraphon((x, y) -> Normal(0, 1), Matrix{Float64})
```

# See Also
- [`DecoratedSBM`](@ref): Block model with distributions
- [`SimpleContinuousGraphon`](@ref): Simple graphon with probabilities
"""
struct DecoratedGraphon{T,M,F,D} <: AbstractGraphon{T,M}
    f::F
end

function DecoratedGraphon(f::F) where {F}
    d = f(0.1, 0.2)
    T = _infer_eltype(d)
    return DecoratedGraphon{T,Matrix{T},F,typeof(d)}(f)
end


function DecoratedGraphon(f::F, ::Type{M}) where {F,M}
    d = f(0.1, 0.2)
    @argcheck _infer_eltype(d) <: eltype(M)
    return DecoratedGraphon{eltype(M),M,F,typeof(d)}(f)
end

(g::DecoratedGraphon)(x, y) = g.f(x, y)


## Decorated Block models

"""
    DecoratedSBM{D,M,P,S,S2} <: AbstractGraphon{eltype(M),M}

A Stochastic Block Model where edges have rich attributes drawn from distributions.
Each block pair (i,j) has an associated distribution from which edge values are sampled.

# Fields
- `θ::P`: K×K matrix of distributions, where θ[i,j] is the distribution for edges between block i and j
- `size::S`: Vector of block sizes (must sum to 1)
- `cumsize::S2`: Cumulative sum of block sizes

# Constructor
    DecoratedSBM(θ, sizes, M=Matrix{...})

Create a decorated SBM with distribution matrix `θ` and block sizes `sizes`.

# Arguments
- `θ`: K×K matrix of Distribution objects
- `sizes`: Vector of K positive values that sum to 1
- `M`: (Optional) Matrix type for sampled graphs

# Examples
```julia
using Distributions

# Two-block model with Normal edges
θ = [Normal(1.0, 0.1) Normal(0.0, 0.1);
     Normal(0.0, 0.1) Normal(1.0, 0.1)]
sizes = [0.5, 0.5]
dsbm = DecoratedSBM(θ, sizes)

# Poisson-weighted edges
θ = [Poisson(10) Poisson(2); Poisson(2) Poisson(10)]
dsbm = DecoratedSBM(θ, sizes)
```

# See Also
- [`DecoratedGraphon`](@ref): Continuous decorated graphon
- [`SBM`](@ref): Simple block model with probabilities
"""
struct DecoratedSBM{D,M,P<:AbstractMatrix{D},S,S2} <: AbstractGraphon{eltype(M),M}
    θ::P
    size::S
    cumsize::S2
end

function DecoratedSBM(θ::AbstractMatrix{D}, sizes, M=Matrix{_infer_eltype(θ[1, 1])}) where {D}
    cumsizes = cumsum(sizes)
    @argcheck last(cumsizes) ≈ 1
    return DecoratedSBM{eltype(θ),M,typeof(θ),typeof(sizes),typeof(cumsizes)}(θ, sizes, cumsizes)
end


function (g::DecoratedSBM)(x, y)
    latents_x = _convert_latent_to_block(g, x)
    latents_y = _convert_latent_to_block(g, y)
    return g.θ[latents_x, latents_y]
end


"""
    empirical_graphon(f::DecoratedGraphon, k::Int) -> DecoratedSBM

Discretize a continuous decorated graphon into a block model with `k` blocks
by evaluating the graphon at a uniform grid of points.

# Arguments
- `f`: A decorated graphon
- `k`: Number of blocks for discretization

# Returns
A [`DecoratedSBM`](@ref) with `k` blocks approximating the continuous graphon.

# Examples
```julia
using Distributions

# Create continuous decorated graphon
f = DecoratedGraphon((x, y) -> Normal(x * y, 0.1))

# Discretize into 10-block model
dsbm = empirical_graphon(f, 10)
```

# See Also
- [`DecoratedGraphon`](@ref): Continuous decorated graphon
- [`DecoratedSBM`](@ref): Block model with distributions
"""
function empirical_graphon(f::DecoratedGraphon{T,M,F,D}, k::Int) where {T,M,F,D}
    ξs = range(0, stop=1, length=k)
    sizes = fill(1 / k, k)
    # dirty hack to ensure sum(sizes) == 1
    sizes[end] += 1 - sum(sizes)
    θ = [f(ξs[i], ξs[j]) for i in 1:k, j in 1:k]
    return DecoratedSBM(θ, sizes, M)
end
