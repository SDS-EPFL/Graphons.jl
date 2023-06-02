struct GraphonFunction{F <: Function} <: AbstractGraphon
    graphon_function::F
    function GraphonFunction(f::F) where {F<:Function}
        # maybe test that f takes two arguments and returns a Float64 ?
        new{typeof(f)}(f)
    end
end

function _probs(s::GraphonFunction, i, j)
    return s.graphon_function(i, j)
end
