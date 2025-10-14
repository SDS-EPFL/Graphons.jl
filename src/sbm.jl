
struct SBM{P,S,S2} <: AbstractGraphon{Bool,BitMatrix}
    θ::P
    size::S
    cumsize::S2
end

function SBM(θ, sizes)
    @argcheck all(x -> x <= 1 && x >= 0, θ)
    @argcheck all(sizes .> 0)
    cumsizes = cumsum(sizes)
    @argcheck last(cumsizes) ≈ 1
    return SBM(θ, sizes, cumsizes)
end


function _convert_latent_to_block(f::SBM, ξ)
    return findfirst(y -> ξ <= y, f.cumsize)
end


function _rand!(rng::AbstractRNG, f::SBM, A::BitMatrix, ξs)
    latents = map(x -> _convert_latent_to_block(f, x), ξs)
    for j in axes(A, 2)
        for i in axes(A, 1)
            if Base.rand(rng) < f.θ[latents[i], latents[j]]
                A[i, j] = one(eltype(A))
                A[j, i] = A[i, j]
            end
        end
    end
    return A
end


function empirical_graphon(f::SimpleContinuousGraphon, k::Int)
    ξs = range(0, stop=1, length=k)
    sizes = fill(1 / k, k)
    return SBM(f.(ξs, ξs), sizes)
end
