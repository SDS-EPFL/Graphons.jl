module SuiteSparseGraphBLASExt

using Random
import SuiteSparseGraphBLAS: AbstractGBArray
import Graphon: rand, _rand!, AbstractGraphon

function rand(rng::AbstractRNG, f::AbstractGraphon{T,M}, n::Int) where {T,M<:AbstractGBArray}
    return _rand!(rng, f, M(n, n), Base.rand(rng, n))
end


function sample(rng::AbstractRNG, f::AbstractGraphon{T,M}, ξs) where {T,M<:AbstractGBArray}
    n = length(ξs)
    return _rand!(rng, f, M(n, n), ξs)
end

end
