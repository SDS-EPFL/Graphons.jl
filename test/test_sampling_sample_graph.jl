using Graphons
using Test
using Random
using Distributions
using LinearAlgebra

@testset "Sampling - sample_graph" begin
    @testset "sample_graph with fixed latents" begin
        rng = MersenneTwister(999)
        f = SimpleContinuousGraphon((x, y) -> x * y)
        ξs = [0.2, 0.5, 0.8]

        A = sample_graph(rng, f, ξs)
        @test A isa BitMatrix
        @test size(A) == (3, 3)
        @test issymmetric(A)
    end

    @testset "sample_graph with n nodes" begin
        rng = MersenneTwister(888)
        sbm = SBM([0.8 0.2; 0.2 0.7], [0.6, 0.4])

        A = sample_graph(rng, sbm, 10)
        @test A isa BitMatrix
        @test size(A) == (10, 10)
        @test issymmetric(A)

        # Test default RNG version
        B = sample_graph(sbm, 10)
        @test B isa BitMatrix
        @test size(B) == (10, 10)
    end

    @testset "sample_graph reproducibility" begin
        rng1 = MersenneTwister(555)
        rng2 = MersenneTwister(555)
        f = SimpleContinuousGraphon((x, y) -> 0.5)
        ξs = range(0, 1, length=5)

        A1 = sample_graph(rng1, f, ξs)
        A2 = sample_graph(rng2, f, ξs)
        @test A1 == A2
    end

    @testset "sample_graph with decorated graphon" begin
        rng = MersenneTwister(777)
        f = DecoratedGraphon((x, y) -> Exponential(1.0))

        A = sample_graph(rng, f, 6)
        @test A isa Matrix{Float64}
        @test size(A) == (6, 6)
        @test issymmetric(A)
        @test all(A .>= 0)  # Exponential distribution is non-negative
    end

    @testset "Small graphs" begin
        f = SimpleContinuousGraphon((x, y) -> 1.0)  # Always connected
        A = rand(f, 2)
        @test size(A) == (2, 2)
        @test A[1, 2] == A[2, 1]
    end
end
