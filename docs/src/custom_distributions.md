# Custom Distribution Types

When using `DecoratedGraphon`, the graphon function returns a distribution
object that is sampled to generate edge values. While Graphons.jl is designed
to work with distributions from `Distributions.jl`, you can use custom
distribution types as long as they implement the required interface.

## Required Methods

A custom distribution type `D` must implement the following methods:

1. **`Base.rand(rng::AbstractRNG, d::D)`** - Sample a single value from the
   distribution
2. **`Base.eltype(d::D)`** - Return the element type of samples from the
   distribution

These are the minimal requirements for a distribution to work with
`DecoratedGraphon`.

## Optional Methods

For better integration with the package, you may also want to implement:

- **`Distributions.params(d::D)`** - Return the parameters of the distribution
  (for display/debugging)
- **`Base.show(io::IO, d::D)`** - Custom string representation

## Example: Custom Discrete Distribution

Here's an example of a simple custom distribution that samples from a fixed set
of values:

```julia
struct CustomCategorical{T}
    values::Vector{T}
    probabilities::Vector{Float64}

    function CustomCategorical(values::Vector{T}, probs::Vector{Float64}) where T
        @assert length(values) == length(probs)
        @assert sum(probs) ≈ 1.0
        @assert all(p >= 0 for p in probs)
        new{T}(values, probs)
    end
end

# Required: sampling method
function Base.rand(rng::AbstractRNG, d::CustomCategorical)
    u = rand(rng)
    cumsum_prob = 0.0
    for (val, prob) in zip(d.values, d.probabilities)
        cumsum_prob += prob
        if u <= cumsum_prob
            return val
        end
    end
    return d.values[end]  # fallback due to floating point
end

# Required: element type
Base.eltype(::CustomCategorical{T}) where T = T

# Optional: display
function Base.show(io::IO, d::CustomCategorical)
    print(io, "CustomCategorical(", d.values, ", ", d.probabilities, ")")
end
```

You can then use this custom distribution with `DecoratedGraphon`:

```julia
using Graphons

# Create a graphon that returns custom distributions
function W_custom(x, y)
    if x + y < 1.0
        return CustomCategorical([1, 2, 3], [0.5, 0.3, 0.2])
    else
        return CustomCategorical([2, 3, 4], [0.2, 0.5, 0.3])
    end
end

graphon = DecoratedGraphon(W_custom)
A = rand(graphon, 50)  # Sample a 50×50 graph
```

For a complete working example with visualization, see the
[Custom Distributions](@ref) tutorial.

## Notes on Type Inference

The `DecoratedGraphon` constructor will automatically infer the edge type by
calling the graphon function at a test point (0.1, 0.2) and checking the
element type of the returned distribution. Make sure your custom distribution's
`eltype` method returns the correct type.

For multivariate distributions (distributions that return vectors), wrap the
result in `StaticArrays.SVector` for better performance:

```julia
Base.eltype(::MyMultivariateDistribution) = SVector{3, Float64}
```
