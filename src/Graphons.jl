module Graphons

using Random
using SparseArrays
using StaticArrays
import Base.rand
using ArgCheck
import Distributions: DiscreteMultivariateDistribution, UnivariateDistribution, MultivariateDistribution


"""
    AbstractGraphon{T,M}

T is the edge type:
 - Bool for a simple graph
 - <:Real for a weighted graph
 - <:AbstractVector{Bool} for a multiplex network
 - ...

M is the type of the representation of the sampled graph from the graphon, e.g. a simple
graph with boolean edges can be represented by
    - a dense matrix of Bool -> M = Matrix{Bool}
    - a sparse matrix of Bool -> M = SparseMatrixCSC{Bool,Int}
    - an adjacency list -> M = Vector{Vector{Int}}

for now we assume that M is such that T <: eltype(M) and that it can be created with
    `make_empty_graph(M, n)` which creates an empty graph of size n x n
"""
abstract type AbstractGraphon{T,M} end

const SimpleGraphon{M} = AbstractGraphon{Bool,M} where {M<:AbstractArray{Bool}}
const WeightedGraphon{M} = AbstractGraphon{Float64,M} where {M<:AbstractArray{Float64}}


include("utils.jl")
include("sampling.jl")
include("simple_graphon.jl")
include("decorated_graphon.jl")



export rand, sample_graph
export SBM, SimpleContinuousGraphon, empirical_graphon
export DecoratedSBM, DecoratedGraphon


end
