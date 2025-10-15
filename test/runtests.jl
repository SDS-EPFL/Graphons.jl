using Graphons
using Test
using Random
using SparseArrays
using StaticArrays
using Distributions
using LinearAlgebra

@testset "Graphons.jl" begin
    include("test_utils.jl")
    include("test_simple_graphon.jl")
    include("test_sbm.jl")
    include("test_decorated_graphon.jl")
    include("test_decorated_sbm.jl")
    include("test_sampling_rand.jl")
    include("test_sampling_sample_graph.jl")
    include("test_integration.jl")
    include("test_graphblas_ext.jl")
end
