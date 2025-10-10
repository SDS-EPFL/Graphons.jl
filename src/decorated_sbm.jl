
struct DecoratedSBM{D,M,P<:AbstractMatrix{D},S,S2} <: AbstractGraphon{eltype(D),M}
    θ::P
    size::S
    cumsize::S2
end

function DecoratedSBM(θ::AbstractMatrix{D}, sizes, M=Matrix{eltype(D)}) where {D}
    #TODO: use @argcheck
    cumsizes = cumsum(sizes)
    @assert last(cumsizes) ≈ 1
    return DecoratedSBM{eltype(θ),M,typeof(θ),typeof(sizes),typeof(cumsizes)}(θ, sizes, cumsizes)
end
