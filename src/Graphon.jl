module Graphon

struct SBM
    θ::Matrix{Float64}
    size::Vector{Float64}
    cumulative_size::Vector{Float64}
end

function SBM(θ::Matrix{Float64}, groupsize::Vector{Float64})
    @assert size(θ, 1) == size(θ, 2)
    @assert length(groupsize) == size(θ, 1)
    @assert all(0 .<= θ .<= 1)
    @assert all(0 .<= groupsize .<= 1)
    @assert sum(groupsize) ≈ 1
    return SBM(θ, groupsize, cumsum(groupsize))
end

function _rand(s::SBM, i, j)
    return Int(rand() < s.θ[findfirst(x -> i <= x, s.cumulative_size),
                   findfirst(x -> j <= x, s.cumulative_size)])
end

function draw(s::SBM, n, latent)
    A = Matrix{Int}(undef, n, n)
    for i in 1:n
        A[i, i] = 0
        for j in (i + 1):n
            A[i, j] = _rand(s, latent[i], latent[j])
            A[j, i] = A[i, j]
        end
    end
    return A
end

function draw_non_exchangeable(s::SBM, n)
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

draw_exchangeable(s::SBM, n) = draw(s, n, rand(Uniform(0, 1), n))
sample(s::SBM, n) = draw_exchangeable(s, n)

export SBM, sample

end
