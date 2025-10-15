## random unobserved latents

rand(f::AbstractGraphon, n::Int) = rand(Random.default_rng(), f, n)


"""
    rand(rng::AbstractRNG, f::AbstractGraphon{T,M}, n::Int) where {T,M}

    Generates a random graph according to the graphon `f` with `n` nodes.
    The latent positions are drawn uniformly at random in [0,1].

    The generated graph is of type `M` and has edge type `T`.

"""
function rand(rng::AbstractRNG, f::AbstractGraphon{T,M}, n::Int) where {T,M}
    return _rand!(rng, f, make_empty_graph(M, n), Base.rand(rng, n))
end

"""
    _rand!(rng::AbstractRNG, f::AbstractGraphon{T,M}, A::M, ξs)

    Generates a random graph according to the graphon `f` and the latent positions `ξs`.
    The generated graph is stored in `A`.

!!! warning
    This function expects that `A` is an empty graph of the right size and type. It does not
    try to clean it up before filling it. See `make_empty_graph` for more details.
"""
function _rand!(rng::AbstractRNG, f::AbstractGraphon{T,M}, A::M, ξs) where {T,M}
    for j in axes(A, 2)
        for i in axes(A, 1)
            if i < j
                A[i, j] = rand(rng, f(ξs[i], ξs[j]))
                A[j, i] = A[i, j]
            end
        end
    end
    return A
end

function _rand!(rng::AbstractRNG, f::SimpleGraphon{M}, A::M, ξs) where {M}
    for j in axes(A, 2)
        for i in axes(A, 1)
            if i < j && Base.rand(rng) < f(ξs[i], ξs[j])
                A[i, j] = one(eltype(A))
                A[j, i] = A[i, j]
            end
        end
    end
    return A
end



## known latents

sample_graph(f::AbstractGraphon, args...) = sample_graph(Random.default_rng(), f, args...)


"""
    sample_graph(rng::AbstractRNG, f::AbstractGraphon, n::Int)

    Generates a random graph according to the graphon `f` with `n` nodes.
    The latent positions are drawn uniformly at random in [0,1].

    The generated graph is of type `M` and has edge type `T`.
"""
function sample_graph(rng::AbstractRNG, f::AbstractGraphon, n::Int)
    return sample_graph(rng, f, range(0, 1, length=n))
end

function sample_graph(rng::AbstractRNG, f::AbstractGraphon{T,M}, ξs) where {T,M}
    n = length(ξs)
    return _rand!(rng, f, make_empty_graph(M, n), ξs)
end
