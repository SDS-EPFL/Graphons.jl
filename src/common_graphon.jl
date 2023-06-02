struct GraphonFunction{F <: Function} <: AbstractGraphon
    graphon_function::F
    function GraphonFunction(f::F) where {F <: Function}
        # maybe test that f takes two arguments and returns a Float64 ?
        new{typeof(f)}(f)
    end
end

function _probs(s::GraphonFunction, i, j)
    return s.graphon_function(i, j)
end

function graphon_function_product(x, y)
    return x * y
end

function graphon_function_exp_07(x, y)
    return exp(-(x^0.7 + y^0.7))
end

function graphon_function_polynomial(x, y)
    return 0.25 * (x^2 + y^2 + sqrt(x) + sqrt(y))
end

function graphon_function_mean(x, y)
    return 0.5 * (x + y)
end

function graphon_function_logit_sum(x, y)
    return 1 / (1 + exp(-10 * (x^2 + y^2)))
end

function graphon_function_latent_distance(x, y)
    return abs(x - y)
end

function graphon_function_logit_max_power(x, y)
    return 1 / (1 + exp(-(max(x, y)^2 + min(x, y)^4)))
end

function graphon_function_exp_max_power(x, y)
    return exp(-max(x, y)^(3 / 4))
end

function graphon_function_exp_polynomial(x, y)
    return exp(-0.5 * (min(x, y) + sqrt(x) + sqrt(y)))
end

function graphon_function_log(x, y)
    return log1p(0.5 * max(x, y))
end
