
struct DecoratedGraphon{T,M,F,D} <: AbstractGraphon{T,M}
    f::F
end

function DecoratedGraphon(f::F) where {F}
    d = f(0.1, 0.2)
    T = _infer_eltype(d)
    return DecoratedGraphon{T,Matrix{T},F,typeof(d)}(f)
end


function _infer_eltype(d::UnivariateDistribution)
    return eltype(d)
end

function _infer_eltype(d::MultivariateDistribution)
    return SizedVector{length(d),eltype(d)}
end

function DecoratedGraphon(f::F, ::Type{M}) where {F,M}
    d = f(0.1, 0.2)
    @argcheck eltype(d) <: eltype(M)
    return DecoratedGraphon{eltype(M),M,F,typeof(d)}(f)
end

(g::DecoratedGraphon)(x, y) = g.f(x, y)

function _rand!(rng::AbstractRNG, f::DecoratedGraphon{T,M}, A::M, ξs) where {T,M}
    for j in axes(A, 2)
        for i in axes(A, 1)
            if i < j
                A[i, j] = rand(rng, f(ξs[i], ξs[j]))
                A[j, i] = A[i, j]
            end
        end
    end
    return A
end
