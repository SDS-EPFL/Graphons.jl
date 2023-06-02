struct SBM <: AbstractGraphon
    θ::Matrix{Float64}
    size::Vector{Float64}
    cumulative_size::Vector{Float64}

    function SBM(θ::Matrix{Float64}, groupsize::Vector{Float64})
        @assert size(θ, 1) == size(θ, 2)
        @assert length(groupsize) == size(θ, 1)
        @assert all(0 .<= θ .<= 1)
        @assert all(0 .<= groupsize .<= 1)
        @assert sum(groupsize) ≈ 1
        new(θ, groupsize, cumsum(groupsize))
    end
end

function _probs(s::SBM, i, j)
    return s.θ[findfirst(x -> i <= x, s.cumulative_size),
               findfirst(x -> j <= x, s.cumulative_size)]
end
