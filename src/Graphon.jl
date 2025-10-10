module Graphon

using Random
using SparseArrays
import SuiteSparseGraphBLAS: AbstractGBArray
import Base.rand

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

for now we assume that M is such that T <: eltype(M) and that it can be created with M(undef, n, n)
"""
abstract type AbstractGraphon{T,M} end

const SimpleGraphon{M} = AbstractGraphon{Bool,M} where {M<:AbstractArray{Bool}}
const WeightedGraphon{M} = AbstractGraphon{Float64,M} where {M<:AbstractArray{Float64}}

rand(f::AbstractGraphon, n::Int) = rand(Random.default_rng(), f, n)

function rand(rng::AbstractRNG, f::AbstractGraphon{T,M}, n::Int) where {T,M}
    return _rand!(rng, f, M(undef, n, n), Base.rand(rng, n))
end




sample(f::AbstractGraphon, n::Int) = sample(Random.default_rng(), f, n)
sample(f::AbstractGraphon, ξ::AbstractVector) = sample(Random.default_rng(), f, ξ)

function sample(rng::AbstractRNG, f::AbstractGraphon, n::Int)
    return sample(rng, f, 0:1/n:1)
end

function sample(rng::AbstractRNG, f::AbstractGraphon{T,M}, ξs) where {T,M}
    n = length(ξs)
    return _rand!(rng, f, M(undef, n, n), ξs)
end


# potentialy move to extension for SuiteSparseGraphBLAS

function rand(rng::AbstractRNG, f::AbstractGraphon{T,M}, n::Int) where {T,M<:AbstractGBArray}
    return _rand!(rng, f, M(n, n), Base.rand(rng, n))
end


function sample(rng::AbstractRNG, f::AbstractGraphon{T,M}, ξs) where {T,M<:AbstractGBArray}
    n = length(ξs)
    return _rand!(rng, f, M(n, n), ξs)
end


include("graphonfunction.jl")
include("sbm.jl")



export rand
export SBM, SimpleContinuousGraphon

end
