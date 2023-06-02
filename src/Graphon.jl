module Graphon

using Distributions, LinearAlgebra


abstract type AbstractGraphon end


function _rand(s::AbstractGraphon, i, j)
    return Int(rand() < _probs(s, i, j))
end

function draw(s::AbstractGraphon, n, latent)
    A = Matrix{Int}(undef, n, n)
    for i in 1:n
        A[i, i] = 0
        for j in (i + 1):n
            A[i, j] = _rand(s, latent[i], latent[j])
            A[j, i] = A[i, j]
        end
    end
    return Symmetric(A)
end

function draw_non_exchangeable(s::AbstractGraphon, n)
    A = Matrix{Int}(undef, n, n)
    for i in 1:n
        A[i, i] = 0
        for j in (i + 1):n
            A[i, j] = _rand(s, i / n, j / n)
            A[j, i] = A[i, j]
        end
    end
    return A
end

draw_exchangeable(s::AbstractGraphon, n) = draw(s, n, rand(Uniform(0, 1), n))
sample(s::AbstractGraphon, n, exchangeable = true) = exchangeable ? draw_exchangeable(s, n) : draw_non_exchangeable(s, n)


include("sbm.jl")
include("common_graphon.jl")

export SBM, sample, GraphonFunction

end
