module MakieExt

using Makie
using Graphons
import Distributions: DiscreteNonParametric, params, Distribution
import Graphons: _convert_latent_to_block, AbstractGraphon

function Makie.convert_arguments(::Type{<:AbstractPlot}, graphon::SimpleContinuousGraphon)
    x = collect(0:0.01:1)
    return (x, x, graphon.f)
end

function Makie.convert_arguments(::Type{<:AbstractPlot}, graphon::SBM)
    x = collect(0:0.01:1)
    ξs = map(x -> _convert_latent_to_block(graphon, x), x)
    return (x, x, graphon.θ[ξs, ξs])
end

function Makie.convert_arguments(
    ::Type{<:AbstractPlot}, graphon::DecoratedGraphon{T,M,F,D},
    k::Int=1) where {T,M,F,D<:Distribution}
    x = collect(0:0.01:1)
    return (x, x, [_extract_param(graphon(xi, yi), k) for xi in x, yi in x])
end


function _extract_param(d::Distribution, k::Int)
    return params(d)[k]
end

function _extract_param(d::DiscreteNonParametric, k::Int)
    return params(d)[2][k]
end

end
