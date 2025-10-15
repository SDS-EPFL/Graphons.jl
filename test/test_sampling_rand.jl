using Graphons
using Test
using Random
using SparseArrays
using Distributions
using LinearAlgebra

@testset "Sampling - rand" begin
    @testset "SimpleContinuousGraphon sampling" begin
        rng = MersenneTwister(123)
        f = SimpleContinuousGraphon((x, y) -> 0.5)

        # Test with default RNG
        A = rand(f, 10)
        @test A isa BitMatrix
        @test size(A) == (10, 10)
        @test issymmetric(A)
        @test all(diag(A) .== 0)  # No self-loops

        # Test with seeded RNG for reproducibility
        rng1 = MersenneTwister(456)
        rng2 = MersenneTwister(456)
        A1 = rand(rng1, f, 5)
        A2 = rand(rng2, f, 5)
        @test A1 == A2
    end

    @testset "SBM sampling" begin
        rng = MersenneTwister(789)
        θ = [0.9 0.1; 0.1 0.9]
        sizes = [0.5, 0.5]
        sbm = SBM(θ, sizes)

        A = rand(rng, sbm, 20)
        @test A isa BitMatrix
        @test size(A) == (20, 20)
        @test issymmetric(A)
        @test all(diag(A) .== 0)
    end

    @testset "DecoratedGraphon sampling" begin
        rng = MersenneTwister(321)
        f = DecoratedGraphon((x, y) -> Normal(x + y, 0.1))

        A = rand(rng, f, 8)
        @test A isa Matrix{Float64}
        @test size(A) == (8, 8)
        @test issymmetric(A)
    end

    @testset "DecoratedSBM sampling" begin
        rng = MersenneTwister(654)
        θ = [Normal(1.0, 0.1) Normal(0.0, 0.1); Normal(0.0, 0.1) Normal(1.0, 0.1)]
        sizes = [0.5, 0.5]
        dsbm = DecoratedSBM(θ, sizes)

        A = rand(rng, dsbm, 12)
        @test A isa Matrix{Float64}
        @test size(A) == (12, 12)
        @test issymmetric(A)
    end

    @testset "Sampling with sparse matrices" begin
        rng = MersenneTwister(111)
        f = SimpleContinuousGraphon((x, y) -> 0.1, SparseMatrixCSC{Bool,Int})

        A = rand(rng, f, 15)
        @test A isa SparseMatrixCSC{Bool,Int}
        @test size(A) == (15, 15)
        @test issymmetric(A)
    end

    @testset "Extreme probabilities" begin
        # Always empty graph
        f_empty = SimpleContinuousGraphon((x, y) -> 0.0)
        A_empty = rand(MersenneTwister(123), f_empty, 10)
        @test all(.!A_empty)

        # Always full graph (except diagonal)
        f_full = SimpleContinuousGraphon((x, y) -> 1.0)
        A_full = rand(MersenneTwister(123), f_full, 10)
        @test sum(A_full) == 10 * 9  # n(n-1) edges for complete graph
    end

    @testset "Symmetry preservation" begin
        rng = MersenneTwister(246)
        f = SimpleContinuousGraphon((x, y) -> abs(x - y))

        A = rand(rng, f, 15)
        @test issymmetric(A)

        # Check no self-loops
        @test all(diag(A) .== 0)
    end
end
