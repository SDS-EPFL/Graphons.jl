struct SimpleContinuousGraphon{M,F} <: SimpleGraphon{M}
    f::F
end

(g::SimpleContinuousGraphon)(x, y) = g.f(x, y)

function SimpleContinuousGraphon(f::F, M=BitMatrix) where {F}
    return SimpleContinuousGraphon{M,F}(f)
end


function _rand!(rng::AbstractRNG, f::SimpleContinuousGraphon{M}, A::M, ξs) where {M}
    fill!(A, false)
    for j in axes(A, 2)
        for i in axes(A, 1)
            if i <= j
                A[i, j] = A[j, i]
            else
                A[i, j] = Base.rand(rng) < f(ξs[i], ξs[j])
            end
        end
    end
    return A
end



function _rand!(rng::AbstractRNG, f::SimpleContinuousGraphon{M}, A::M, ξs) where {M<:SparseArrays.AbstractSparseMatrixCSC}
    A.rowval .= zero(eltype(A))
    dropzeros!(A)
    for j in axes(A, 2)
        for i in axes(A, 1)
            if i <= j
                A[i, j] = A[j, i]
            else
                A[i, j] = Base.rand(rng) < f(ξs[i], ξs[j])
            end
        end
    end
    return A
end
