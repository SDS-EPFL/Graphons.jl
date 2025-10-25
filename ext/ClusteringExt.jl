module ClusteringExt

using Graphons
using ArgCheck
using Clustering
using Random
import Distributions: support, DiscreteNonParametric, params, logpdf

import Graphons: SSM, upper_triangular_to_full, matrix_type, _extract_param,
                 _convert_latent_to_block, SimpleGraphon, AbstractGraphon, estimate_ssm

export SSM, estimate_ssm

function SSM(sbm::Union{SBM, DecoratedSBM}, num_shapes::Int,
        rng::AbstractRNG = Random.default_rng())
    @argcheck num_shapes>0 "Number of shapes must be positive."
    K = length(sbm.size)
    @assert num_shapes<K * (K + 1) / 2 "Number of shapes must be less than number of unique block pairs."

    # k-means++
    X = _extract_params_triu(sbm)
    θ, block_pair_to_shape = _ssm_params(X, num_shapes, sbm, rng)
    return Graphons.SSM(θ, block_pair_to_shape, sbm.size, sbm.cumsize, matrix_type(sbm))
end

function estimate_ssm(sbm::Union{SBM, DecoratedSBM}, A, latents, shape_range,
        rng::AbstractRNG = Random.default_rng())
    best_ssm = nothing
    best_criterion_value = Inf
    criterion_values = zeros(length(shape_range))
    X = _extract_params_triu(sbm)
    for (i, num_shapes) in enumerate(shape_range)
        ssm = SSM(_ssm_params(X, num_shapes, sbm, rng)..., sbm.size, sbm.cumsize)
        criterion_value = bic(ssm, A, latents)
        criterion_values[i] = criterion_value
        if criterion_value < best_criterion_value
            best_criterion_value = criterion_value
            best_ssm = ssm
        end
    end
    return best_ssm, criterion_values
end

function SSM(sbm::Union{SBM, DecoratedSBM}, A, latents,
        shape_range, rng::AbstractRNG = Random.default_rng())
    first(estimate_ssm(sbm, A, latents, shape_range, rng))
end

@inline function logprob(model::SimpleGraphon, A, ξ_i, ξ_j)
    p = model(ξ_i, ξ_j)
    return A * log(p + eps()) + (1 - A) * log(1 - p + eps())
end

logprob(model::AbstractGraphon, A, ξ_i, ξ_j) = logpdf(model(ξ_i, ξ_j), A)

num_params_per_shape(::SimpleGraphon) = 1
num_params_per_shape(model::AbstractGraphon) = length(_extract_param(first(model.θ)))
function num_shapes(model::Union{SBM, DecoratedSBM})
    num_blocks = size(model.θ, 1)
    return num_blocks * (num_blocks + 1) ÷ 2
end
num_shapes(model::Graphons.SSM) = length(model.θ)

function bic(model::Union{SBM, DecoratedSBM, SSM}, A, latents)
    ll = 0.0
    n = length(latents)
    @inbounds for j in eachindex(latents)
        for i in eachindex(latents)
            if i < j
                ll += logprob(model, A[i, j], latents[i], latents[j])
            end
        end
    end
    return -2 * ll + num_params_per_shape(model) * num_shapes(model) * log(n * (n - 1) / 2)
end

## =========================================================================================
## Helper functions for shape models estimation from Block models
## =========================================================================================

function _ssm_params(X, num_shapes, sbm, rng::AbstractRNG = Random.default_rng())
    res = kmeans(X, num_shapes, init = :kmpp, rng = rng)
    block_pair_to_shape = upper_triangular_to_full(assignments(res))
    θ = convert_to_params(res.centers, sbm)
    return θ, block_pair_to_shape
end

function _extract_params_triu(sbm::SBM)
    K = length(sbm.size)
    return reduce(hcat, sbm.θ[row, col] for col in 1:K for row in 1:col)
end

function _extract_params_triu(dsbm::DecoratedSBM)
    K = length(dsbm.size)
    return reduce(
        hcat, Graphons._extract_param(dsbm.θ[row, col], :) for col in 1:K for row in 1:col)
end

function convert_to_params(centers, ::SBM)
    return [centers[i] for i in axes(centers, 2)]
end

function convert_to_params(centers, ::DecoratedSBM{D}) where {D}
    return [D(centers[:, i]...) for i in axes(centers, 2)]
end

# specialization for DiscreteNonParametric as it requires support to be specified
function convert_to_params(centers,
        sbm::DecoratedSBM{DiscreteNonParametric{T, P, Ts, Ps}}) where {T, P, Ts, Ps}
    s = support(sbm.θ[1, 1])
    return [DiscreteNonParametric{T, P, Ts, Ps}(s, convert(Ps, centers[:, i]))
            for i in axes(centers, 2)]
end

end
