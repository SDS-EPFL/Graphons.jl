module MakieExt

using Makie
using Graphons
import StatsAPI: params
import Distributions: DiscreteNonParametric, Distribution
import Graphons: _convert_latent_to_block, AbstractGraphon, _extract_param, AbstractGraphon,
                 SimpleGraphon

## Bernoulli graphons

function Makie.used_attributes(::Type{<:Plot}, graphon::Union{SimpleContinuousGraphon, SBM})
    return (:res,)
end

function Makie.convert_arguments(
        ::Type{<:AbstractPlot}, graphon::SimpleGraphon; res = 0.01)
    x = collect(0:res:1)
    return (x, x, [graphon(x, y) for x in x, y in x])
end

function Makie.convert_arguments(::Type{<:AbstractPlot}, graphon::SBM; res = 0.01)
    x = collect(0:res:1)
    ξs = map(x -> _convert_latent_to_block(graphon, x), x)
    return (x, x, graphon.θ[ξs, ξs])
end

## Decorated graphons

function Makie.used_attributes(
        ::Type{<:Plot}, graphon::Union{DecoratedGraphon, DecoratedSBM})
    return (:k, :res)
end

function Makie.convert_arguments(
        ::Type{<:AbstractPlot}, graphon::AbstractGraphon; k::Int = 1, res = 0.01)
    x = collect(0:res:1)
    return (x, x, [_extract_param(graphon(xi, yi), k) for xi in x, yi in x])
end

end
