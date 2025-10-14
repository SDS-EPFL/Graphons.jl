
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
    fill!(A, false)
    for j in axes(A, 2)
        for i in axes(A, 1)
            if i <= j
                A[i, j] = A[j, i]
            else
                A[i, j] = Base.rand(rng) < f.θ[latents[i], latents[j]]
            end
        end
    end
    return A
end
