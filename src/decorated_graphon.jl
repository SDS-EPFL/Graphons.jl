

_infer_eltype(d) = eltype(d)

function _infer_eltype(d::MultivariateDistribution)
    return SizedVector{length(d),eltype(d)}
end


## Continuous decorated graphons
struct DecoratedGraphon{T,M,F,D} <: AbstractGraphon{T,M}
    f::F
end

function DecoratedGraphon(f::F) where {F}
    d = f(0.1, 0.2)
    T = _infer_eltype(d)
    return DecoratedGraphon{T,Matrix{T},F,typeof(d)}(f)
end


function DecoratedGraphon(f::F, ::Type{M}) where {F,M}
    d = f(0.1, 0.2)
    @argcheck _infer_eltype(d) <: eltype(M)
    return DecoratedGraphon{eltype(M),M,F,typeof(d)}(f)
end

(g::DecoratedGraphon)(x, y) = g.f(x, y)


## Decorated Block models
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


function (g::DecoratedSBM)(x, y)
    latents_x = _convert_latent_to_block(g, x)
    latents_y = _convert_latent_to_block(g, y)
    return g.θ[latents_x, latents_y]
end


function empirical_graphon(f::DecoratedGraphon{T,M,F,D}, k::Int) where {T,M,F,D}
    ξs = range(0, stop=1, length=k)
    sizes = fill(1 / k, k)
    # dirty hack to ensure sum(sizes) == 1
    sizes[end] += 1 - sum(sizes)
    θ = [f(ξs[i], ξs[j]) for i in 1:k, j in 1:k]
    return DecoratedSBM(θ, sizes, M)
end
