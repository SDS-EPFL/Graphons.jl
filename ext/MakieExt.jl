module MakieExt

using Makie
using Graphons
import StatsAPI: params
import Distributions: DiscreteNonParametric, Distribution
import Graphons: _convert_latent_to_block, AbstractGraphon


## Bernoulli graphons

function Makie.used_attributes(::Type{<:Plot}, graphon::Union{SimpleContinuousGraphon,SBM})
    return (:res,)
end

function Makie.convert_arguments(::Type{<:AbstractPlot}, graphon::SimpleContinuousGraphon; res=0.01)
    x = collect(0:res:1)
    return (x, x, graphon.f)
end

function Makie.convert_arguments(::Type{<:AbstractPlot}, graphon::SBM; res=0.01)
    x = collect(0:res:1)
    ξs = map(x -> _convert_latent_to_block(graphon, x), x)
    return (x, x, graphon.θ[ξs, ξs])
end

## Decorated graphons

function Makie.used_attributes(::Type{<:Plot}, graphon::Union{DecoratedGraphon,DecoratedSBM})
    return (:k, :res,)
end


function Makie.convert_arguments(
    ::Type{<:AbstractPlot}, graphon::Union{DecoratedGraphon,DecoratedSBM}; k::Int=1, res=0.01)
    x = collect(0:res:1)
    return (x, x, [_extract_param(graphon(xi, yi), k) for xi in x, yi in x])
end


## Helper function to extract the k-th parameter from a distribution

function _extract_param(d, k::Int)
    return params(d)[k]
end

function _extract_param(d::DiscreteNonParametric, k::Int)
    return params(d)[2][k]
end

end
