## Helper function to extract the k-th parameter from a distribution

function _extract_param(d, k)
    return collect(Float64, Iterators.flatten(params(d)))[k]
end

function _extract_param(d::DiscreteNonParametric, k)
    return probs(d)[k]
end

_extract_param(d) = _extract_param(d, :)
