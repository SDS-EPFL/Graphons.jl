module Graphons

using Random
using SparseArrays
import Base.rand
using ArgCheck

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

rand(f::AbstractGraphon, n::Int) = rand(Random.default_rng(), f, n)

function rand(rng::AbstractRNG, f::AbstractGraphon{T,M}, n::Int) where {T,M}
    return _rand!(rng, f, make_empty_graph(M, n), Base.rand(rng, n))
end

"""
    _rand!(rng::AbstractRNG, f::AbstractGraphon{T,M}, A::M, ξs)

!!! warning
    This function expects that `A` is an empty graph of the right size and type. It does not
    try to clean it up before filling it. See `make_empty_graph` for more details.
"""
function _rand!(rng::AbstractRNG, f::AbstractGraphon, A, ξs) end


sample(f::AbstractGraphon, n::Int) = sample(Random.default_rng(), f, n)
sample(f::AbstractGraphon, ξ::AbstractVector) = sample(Random.default_rng(), f, ξ)

function sample(rng::AbstractRNG, f::AbstractGraphon, n::Int)
    return sample(rng, f, 0:1/n:1)
end

function sample(rng::AbstractRNG, f::AbstractGraphon{T,M}, ξs) where {T,M}
    n = length(ξs)
    return _rand!(rng, f, make_empty_graph(M, n), ξs)
end


include("utils.jl")
include("graphon_function.jl")
include("sbm.jl")
include("decorated_sbm.jl")




export rand, sample
export SBM, SimpleContinuousGraphon
export DecoratedSBM

end
