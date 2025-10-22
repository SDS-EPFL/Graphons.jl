"""
    SSM{T,M,V,B,S,S2} <: AbstractGraphon{T,M}

A Stochastic Shape Model (SSM) is an extension of the Stochastic Block Model (SBM). See
[verdeyme_hybrid_2024](@cite) for details.
"""
struct SSM{T, M, V, B, S, S2} <: AbstractGraphon{T, M}
    θ::V
    block_pair_to_shape::B
    size::S
    cumsize::S2

    function SSM(
            θ::AbstractVector{<:Real}, block_pair_to_shape, sizes, cumsizes, M = BitMatrix)
        @argcheck size(block_pair_to_shape, 1) == size(block_pair_to_shape, 2)
        @argcheck length(sizes) == size(block_pair_to_shape, 1)
        @argcheck all(sizes .> 0)
        @argcheck last(cumsizes) ≈ 1
        return new{Bool, M, typeof(θ), typeof(block_pair_to_shape),
            typeof(sizes), typeof(cumsizes)}(θ, block_pair_to_shape, sizes, cumsizes)
    end

    function SSM(θ::AbstractVector{D}, block_pair_to_shape,
            sizes, cumsizes, M = Matrix{_infer_eltype(first(θ))}) where {D}
        @argcheck size(block_pair_to_shape, 1) == size(block_pair_to_shape, 2)
        @argcheck length(sizes) == size(block_pair_to_shape, 1)
        @argcheck all(sizes .> 0)
        @argcheck last(cumsizes) ≈ 1
        return new{eltype(D), M, typeof(θ),
            typeof(block_pair_to_shape), typeof(sizes), typeof(cumsizes)}(
            θ, block_pair_to_shape, sizes, cumsizes)
    end
end

function (g::SSM)(x, y)
    latents_x = _convert_latent_to_block(g, x)
    latents_y = _convert_latent_to_block(g, y)
    return g.θ[g.block_pair_to_shape[latents_x, latents_y]]
end

function get_theta_matrix(ssm::SSM)
    K = length(ssm.size)
    θ_matrix = Array{eltype(ssm.θ), 2}(undef, K, K)
    for i in 1:K
        for j in 1:K
            θ_matrix[i, j] = ssm.θ[ssm.block_pair_to_shape[i, j]]
        end
    end
    return θ_matrix
end

function convert_to_sbm(ssm::SSM{Bool, M}) where {M}
    K = length(ssm.size)
    D = eltype(ssm.θ)
    θ_matrix = Array{D, 2}(undef, K, K)
    for i in 1:K
        for j in 1:K
            θ_matrix[i, j] = ssm.θ[ssm.block_pair_to_shape[i, j]]
        end
    end
    return SBM(θ_matrix, ssm.size, M)
end

function convert_to_sbm(ssm::SSM{T, M, V, B, S, S2}) where {T, M, V, B, S, S2}
    K = length(ssm.size)
    D = eltype(ssm.θ)
    θ_matrix = Array{D, 2}(undef, K, K)
    for i in 1:K
        for j in 1:K
            θ_matrix[i, j] = ssm.θ[ssm.block_pair_to_shape[i, j]]
        end
    end
    return DecoratedSBM(θ_matrix, ssm.size, M)
end
