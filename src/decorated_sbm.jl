
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



function _rand!(rng::AbstractRNG, f::DecoratedSBM{D,M}, A::M, ξs) where {D,M}
    latents = map(x -> findfirst(y -> x <= y, f.cumsize), ξs)
    fill!(A, zero(eltype(A)))
    for j in axes(A, 2)
        for i in axes(A, 1)
            if i <= j
                A[i, j] = A[j, i]
            else
                A[i, j] = rand(rng, f.θ[latents[i], latents[j]])
            end
        end
    end
    return A
end
