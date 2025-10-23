## default empty graph

function make_empty_graph(::Type{M}, n) where {M <: Matrix}
    return zeros(eltype(M), n, n)
end

# default slow fallback

function make_empty_graph(::Type{M}, n) where {M <: AbstractMatrix}
    A = Array{eltype(M), 2}(undef, n, n)
    fill!(A, zero(eltype(M)))
    return A
end

function _convert_latent_to_block(sbm, ξ)
    res = findfirst(y -> ξ <= y, sbm.cumsize)
    if isnothing(res)
        return length(sbm.cumsize)
    else
        return res
    end
end

@inline function _convert_latent_to_block(sbm, ξ1, ξ2)
    index_1 = 0
    index_2 = 0
    @inbounds for index in eachindex(sbm.cumsize)
        if ξ1 <= sbm.cumsize[index] && index_1 == 0
            index_1 = index
        end
        if ξ2 <= sbm.cumsize[index] && index_2 == 0
            index_2 = index
        end
        if index_1 != 0 && index_2 != 0
            break
        end
    end
    return index_1, index_2
end

function upper_triangular_to_full(v::AbstractVector{T}) where {T}
    n = length(v)
    s = Int((sqrt(8n + 1) - 1) / 2)
    s * (s + 1) / 2 == n || error("vec2utri: length of vector is not triangular")
    return [i <= j ? v[j * (j - 1) ÷ 2 + i] : v[i * (i - 1) ÷ 2 + j] for i in 1:s, j in 1:s]
end

## Helper function to extract the k-th parameter from a distribution

function _extract_param(d, k)
    return collect(Float64, Iterators.flatten(params(d)))[k]
end

function _extract_param(d::DiscreteNonParametric, k)
    return params(d)[2][k]
end

_extract_param(d) = _extract_param(d, :)

# specializations

function make_empty_graph(::Type{BitMatrix}, n)
    return falses(n, n)
end

function make_empty_graph(::Type{M}, n) where {M <: SparseArrays.AbstractSparseMatrixCSC}
    return spzeros(eltype(M), n, n)
end

## default clearing of graph
function clear_graph!(A::AbstractMatrix)
    fill!(A, zero(eltype(A)))
end

function clear_graph!(A::SparseArrays.AbstractSparseMatrixCSC)
    droptol!(A, Inf)
end
