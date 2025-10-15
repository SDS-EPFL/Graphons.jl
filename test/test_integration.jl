using Graphons
using Test
using Random
using Distributions
using LinearAlgebra

@testset "Integration tests" begin
    @testset "Full workflow: continuous to empirical to sampling" begin
        # Create continuous graphon
        f = SimpleContinuousGraphon((x, y) -> min(x, y))

        # Discretize to SBM
        sbm = empirical_graphon(f, 4)
        @test sbm isa SBM

        # Sample graph
        rng = MersenneTwister(999)
        A = rand(rng, sbm, 20)
        @test A isa BitMatrix
        @test issymmetric(A)
    end

    @testset "Decorated workflow" begin
        # Create decorated graphon
        f = DecoratedGraphon((x, y) -> Poisson(5 * x * y))

        # Discretize
        dsbm = empirical_graphon(f, 3)
        @test dsbm isa DecoratedSBM

        # Sample
        rng = MersenneTwister(888)
        A = rand(rng, dsbm, 15)
        @test size(A) == (15, 15)
        @test issymmetric(A)
    end

    @testset "Type stability checks" begin
        @testset "make_empty_graph" begin
            @inferred Graphons.make_empty_graph(BitMatrix, 10)
            @inferred Graphons.make_empty_graph(Matrix{Float64}, 10)
        end
    end
end
