module Graphons

using Random
using SparseArrays
using StaticArrays
import Base.rand
using ArgCheck
import Distributions: DiscreteMultivariateDistribution, UnivariateDistribution, MultivariateDistribution


"""
    AbstractGraphon{T,M}

Abstract base type for all graphon models.

# Type Parameters
- `T`: The edge type (e.g., `Bool` for simple graphs, `Float64` for weighted graphs,
  `SizedVector{2,Bool}` for multiplex networks)
- `M`: The matrix type for sampled graphs (e.g., `BitMatrix`, `Matrix{T}`,
  `SparseMatrixCSC{T,Int}`, `GBMatrix{T}`)

# Edge Types
- `Bool`: Simple (unweighted) graphs
- `<:Real`: Weighted graphs with numeric edge weights
- `<:AbstractVector{Bool}`: Multiplex networks with multiple edge types

# Matrix Types
- `BitMatrix` or `Matrix{Bool}`: Dense representation for simple graphs
- `Matrix{T}`: Dense representation for weighted/decorated graphs
- `SparseMatrixCSC{T,Int}`: Sparse representation for low-density graphs
- `GBMatrix{T}`: GraphBLAS representation (requires SuiteSparseGraphBLAS extension)

All graphon types must:
1. Be callable: `graphon(x, y)` returns probability or distribution at positions `x, y ∈ [0,1]`
2. Support `make_empty_graph(M, n)` for creating empty adjacency matrices
3. Implement sampling through `_rand!` method

# Subtypes
- [`SimpleContinuousGraphon`](@ref): Continuous function-based graphons
- [`SBM`](@ref): Stochastic block models
- [`DecoratedGraphon`](@ref): Graphons with distribution-valued edges
- [`DecoratedSBM`](@ref): Block models with distribution-valued edges

# See Also
- [`rand`](@ref), [`sample_graph`](@ref): Sampling functions
"""
abstract type AbstractGraphon{T,M} end

"""
    SimpleGraphon{M}

Type alias for graphons representing simple (unweighted) graphs with Boolean edges.

Equivalent to `AbstractGraphon{Bool,M}` where `M` is a matrix type storing Boolean values.
"""
const SimpleGraphon{M} = AbstractGraphon{Bool,M} where {M<:AbstractArray{Bool}}

"""
    WeightedGraphon{M}

Type alias for graphons representing weighted graphs with Float64 edges.

Equivalent to `AbstractGraphon{Float64,M}` where `M` is a matrix type storing Float64 values.
"""
const WeightedGraphon{M} = AbstractGraphon{Float64,M} where {M<:AbstractArray{Float64}}


include("utils.jl")
include("sampling.jl")
include("simple_graphon.jl")
include("decorated_graphon.jl")



export rand, sample_graph
export SBM, SimpleContinuousGraphon, empirical_graphon
export DecoratedSBM, DecoratedGraphon


end
