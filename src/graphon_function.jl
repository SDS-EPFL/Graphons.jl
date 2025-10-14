struct SimpleContinuousGraphon{M,F} <: SimpleGraphon{M}
    f::F
end

(g::SimpleContinuousGraphon)(x, y) = g.f(x, y)

function SimpleContinuousGraphon(f::F, M=BitMatrix) where {F}
    return SimpleContinuousGraphon{M,F}(f)
end


function _rand!(rng::AbstractRNG, f::SimpleContinuousGraphon{M}, A::M, ξs) where {M}
    for j in axes(A, 2)
        for i in axes(A, 1)
            if Base.rand(rng) < f(ξs[i], ξs[j])
                A[i, j] = one(eltype(A))
                A[j, i] = A[i, j]
            end
        end
    end
    return A
end
