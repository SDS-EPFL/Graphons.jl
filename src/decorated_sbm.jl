
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


function _rand!(rng::AbstractRNG, f::DecoratedSBM{D,M}, A::M, ξs) where {D,M}
    latents = map(x -> findfirst(y -> x <= y, f.cumsize), ξs)
    for j in axes(A, 2)
        for i in axes(A, 1)
            if i < j
                A[i, j] = rand(rng, f.θ[latents[i], latents[j]])
                A[j, i] = A[i, j]
            end
        end
    end
    return A
end



function empirical_graphon(f::DecoratedGraphon{T,M,F,D}, k::Int) where {T,M,F,D}
    ξs = range(0, stop=1, length=k)
    sizes = fill(1 / k, k)
    # dirty hack to ensure sum(sizes) == 1
    sizes[end] += 1 - sum(sizes)
    θ = [f(ξs[i], ξs[j]) for i in 1:k, j in 1:k]
    return DecoratedSBM(θ, sizes, M)
end
