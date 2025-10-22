module ClusteringExt

using Graphons
using ArgCheck
using Clustering
import Distributions: support, DiscreteNonParametric, params

import Graphons: SSM, upper_triangular_to_full, matrix_type

function SSM(sbm::Union{SBM, DecoratedSBM}, num_shapes::Int)
    @argcheck num_shapes>0 "Number of shapes must be positive."
    K = length(sbm.size)
    @assert num_shapes<K * (K + 1) / 2 "Number of shapes must be less than number of unique block pairs."

    # k-means++
    X = _extract_params_triu(sbm)
    θ, block_pair_to_shape = _ssm_params(X, num_shapes, sbm)
    return Graphons.SSM(θ, block_pair_to_shape, sbm.size, sbm.cumsize, matrix_type(sbm))
end

function SSM(sbm::Union{SBM, DecoratedSBM}, data, shape_range, criterion_to_minimize)
    best_ssm = nothing
    best_criterion_value = Inf
    X = _extract_params_triu(sbm)
    for num_shapes in shape_range
        ssm = SSM(_ssm_params(X, num_shapes, sbm)..., sbm.size, sbm.cumsize)
        criterion_value = criterion_to_minimize(ssm, data)
        if criterion_value < best_criterion_value
            best_criterion_value = criterion_value
            best_ssm = ssm
        end
    end
    return best_ssm
end

## =========================================================================================
## Helper functions for shape models estimation from Block models
## =========================================================================================

function _ssm_params(X, num_shapes, sbm)
    res = kmeans(X, num_shapes, init = :kmpp)
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
