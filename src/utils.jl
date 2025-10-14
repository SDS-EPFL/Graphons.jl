## default empty graph

function make_empty_graph(::Type{M}, n) where {M<:Matrix}
    return zeros(eltype(M), n, n)
end

# default slow fallback

function make_empty_graph(::Type{M}, n) where {M<:AbstractMatrix}
    A = Array{eltype(M),2}(undef, n, n)
    fill!(A, zero(eltype(M)))
    return A
end


# specializations

function make_empty_graph(::Type{BitMatrix}, n)
    return falses(n, n)
end

function make_empty_graph(::Type{M}, n) where {M<:SparseArrays.AbstractSparseMatrixCSC}
    return spzeros(eltype(M), n, n)
end


## default clearing of graph
function clear_graph!(A::AbstractMatrix)
    fill!(A, zero(eltype(A)))
end

function clear_graph!(A::SparseArrays.AbstractSparseMatrixCSC)
    droptol!(A, Inf)
end
