module ClusteringExt

using Graphons
using ArgCheck
using Clustering
using Random
using SpecialFunctions: gamma, loggamma
import Distributions: support, DiscreteNonParametric, params, logpdf

import Graphons: SSM, upper_triangular_to_full, matrix_type, _extract_param,
                 _convert_latent_to_block, SimpleGraphon, AbstractGraphon, estimate_ssm,
                 _extract_params_triu, convert_to_params, node_labels_to_latents

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
        rng::AbstractRNG = Random.default_rng(), criterion_f = bic)
    if all(x -> x >= 1, latents)
        @debug "Converting node labels to latent positions."
        if any(x -> x > num_blocks(sbm), latents)
            throw(ArgumentError("latents must be either latent positions in [0,1] or
            integer node labels in {1,2,..., $(num_blocks(sbm))}."))
        end
        latents_checked = node_labels_to_latents(latents, sbm)
    else
        latents_checked = latents
    end
    best_ssm = nothing
    best_criterion_value = Inf
    criterion_values = zeros(length(shape_range))
    X = _extract_params_triu(sbm)
    for (i, num_shapes) in enumerate(shape_range)
        ssm = SSM(_ssm_params(X, num_shapes, sbm, rng)..., sbm.size, sbm.cumsize)
        criterion_value = criterion_f(ssm, A, latents_checked)
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
num_blocks(model::Union{SBM, DecoratedSBM}) = size(model.θ, 1)
num_blocks(model::Graphons.SSM) = length(model.size)

function _ll(model::Union{SBM, DecoratedSBM, SSM}, A, latents)
    ll = 0.0
    @inbounds for j in eachindex(latents)
        for i in eachindex(latents)
            if i < j
                ll += logprob(model, A[i, j], latents[i], latents[j])
            end
        end
    end
    return ll
end

function bic(model::Union{SBM, DecoratedSBM, SSM}, A, latents)
    ll = _ll(model, A, latents)
    n = size(A, 1)
    return -2 * ll + num_params_per_shape(model) * num_shapes(model) * log(n * (n - 1) / 2)
end

function mdl(model::Union{SBM, DecoratedSBM, SSM}, A, latents)
    ll = _ll(model, A, latents)
    if !isfinite(ll)
        @warn "Log-likelihood is not finite"
    end
    n_nodes = size(A, 1)
    n_edges = n_nodes * (n_nodes - 1) ÷ 2
    s = num_shapes(model)
    k = num_blocks(model)
    partition_cost = log_binomial(n_nodes - 1, k - 1) + loggamma(n_nodes + 1) -
                     sum(loggamma(ceil(Int, c * n_nodes) + 1) for c in model.size)
    edge_agg_cost = log_binomial(s + n_edges - 1, n_edges)
    # parameters
    return -ll + partition_cost + edge_agg_cost
end

log_binomial(n, k) = loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)

## =========================================================================================
## Helper functions for shape models estimation from Block models
## =========================================================================================

function _ssm_params(X, num_shapes, sbm, rng::AbstractRNG = Random.default_rng())
    res = kmeans(X, num_shapes, init = :kmpp, rng = rng)
    block_pair_to_shape = upper_triangular_to_full(assignments(res))
    θ = convert_to_params(res.centers, sbm)
    return θ, block_pair_to_shape
end

end
