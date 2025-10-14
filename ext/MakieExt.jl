module MakieExt

using Makie
using Graphons
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


end
