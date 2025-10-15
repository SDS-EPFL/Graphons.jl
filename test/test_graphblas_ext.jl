using Graphons
using Test
using Random
using LinearAlgebra

using SuiteSparseGraphBLAS

@testset "SuiteSparseGraphBLAS Extension" begin
    @testset "make_empty_graph" begin
        # Test GBMatrix creation with Bool
        A = Graphons.make_empty_graph(GBMatrix{Bool}, 5)
        @test A isa GBMatrix{Bool}
        @test size(A) == (5, 5)
        @test nnz(A) == 0

        # Test GBMatrix creation with Float64
        B = Graphons.make_empty_graph(GBMatrix{Float64}, 4)
        @test B isa GBMatrix{Float64}
        @test size(B) == (4, 4)
        @test nnz(B) == 0

        # Test GBMatrix creation with Int
        C = Graphons.make_empty_graph(GBMatrix{Int}, 3)
        @test C isa GBMatrix{Int}
        @test size(C) == (3, 3)
        @test nnz(C) == 0
    end

    @testset "clear_graph!" begin
        # Create a non-empty GBMatrix
        A = GBMatrix{Bool}(5, 5)
        A[1, 2] = true
        A[2, 3] = true
        A[3, 4] = true
        @test nnz(A) > 0

        # Clear it
        Graphons.clear_graph!(A)
        @test nnz(A) == 0

        # Test with Float64
        B = GBMatrix{Float64}(4, 4)
        B[1, 1] = 1.5
        B[2, 3] = 2.7
        @test nnz(B) > 0

        Graphons.clear_graph!(B)
        @test nnz(B) == 0
    end

    @testset "SimpleContinuousGraphon with GBMatrix" begin
        rng = MersenneTwister(123)
        f = SimpleContinuousGraphon((x, y) -> 0.5, GBMatrix{Bool})

        # Test sampling
        A = rand(rng, f, 10)
        @test A isa GBMatrix{Bool}
        @test size(A) == (10, 10)
        @test issymmetric(Matrix(A))
        @test all(diag(Matrix(A)) .== 0)  # No self-loops
    end

    @testset "SBM with GBMatrix" begin
        rng = MersenneTwister(456)
        θ = [0.8 0.2; 0.2 0.7]
        sizes = [0.5, 0.5]

        # Create SBM that uses GBMatrix
        sbm_continuous = SimpleContinuousGraphon((x, y) -> begin
                latent_x = x <= 0.5 ? 1 : 2
                latent_y = y <= 0.5 ? 1 : 2
                θ[latent_x, latent_y]
            end, GBMatrix{Bool})

        A = rand(rng, sbm_continuous, 20)
        @test A isa GBMatrix{Bool}
        @test size(A) == (20, 20)
        @test issymmetric(Matrix(A))
    end

    @testset "sample_graph with GBMatrix" begin
        rng = MersenneTwister(789)
        f = SimpleContinuousGraphon((x, y) -> x * y, GBMatrix{Bool})

        # Test with fixed latents
        ξs = [0.2, 0.5, 0.8]
        A = sample_graph(rng, f, ξs)
        @test A isa GBMatrix{Bool}
        @test size(A) == (3, 3)
        @test issymmetric(Matrix(A))

        # Test with n nodes
        B = sample_graph(rng, f, 8)
        @test B isa GBMatrix{Bool}
        @test size(B) == (8, 8)
        @test issymmetric(Matrix(B))
    end

    @testset "Reproducibility with GBMatrix" begin
        f = SimpleContinuousGraphon((x, y) -> 0.6, GBMatrix{Bool})

        rng1 = MersenneTwister(111)
        rng2 = MersenneTwister(111)

        A1 = rand(rng1, f, 6)
        A2 = rand(rng2, f, 6)

        @test Matrix(A1) == Matrix(A2)
    end

    @testset "Sparse graphs with GBMatrix" begin
        rng = MersenneTwister(222)

        # Low probability - should result in sparse graph
        f_sparse = SimpleContinuousGraphon((x, y) -> 0.05, GBMatrix{Bool})
        A_sparse = rand(rng, f_sparse, 50)

        @test A_sparse isa GBMatrix{Bool}
        @test size(A_sparse) == (50, 50)
        # Should be sparse (less than 20% of possible edges)
        @test nnz(A_sparse) < 50 * 49 * 0.2

        # High probability - should result in dense graph
        f_dense = SimpleContinuousGraphon((x, y) -> 0.95, GBMatrix{Bool})
        A_dense = rand(rng, f_dense, 50)

        @test A_dense isa GBMatrix{Bool}
        @test size(A_dense) == (50, 50)
        # Should be dense (more than 80% of possible edges)
        @test nnz(A_dense) > 50 * 49 * 0.8
    end

    @testset "DecoratedGraphon with GBMatrix" begin
        using Distributions

        rng = MersenneTwister(333)

        # Create a decorated graphon that returns continuous values
        f = DecoratedGraphon((x, y) -> Normal(x + y, 0.1), GBMatrix{Float64})

        A = rand(rng, f, 8)
        @test A isa GBMatrix{Float64}
        @test size(A) == (8, 8)
        @test issymmetric(Matrix(A))

        # Test with Poisson distribution (returns integers)
        g = DecoratedGraphon((x, y) -> Poisson(5 * x * y), GBMatrix{Int})
        B = rand(rng, g, 6)
        @test B isa GBMatrix{Int}
        @test size(B) == (6, 6)
        @test issymmetric(Matrix(B))
    end

    @testset "Edge cases with GBMatrix" begin
        # Very small graph
        f_small = SimpleContinuousGraphon((x, y) -> 1.0, GBMatrix{Bool})
        A_small = rand(MersenneTwister(444), f_small, 2)
        @test size(A_small) == (2, 2)
        @test A_small isa GBMatrix{Bool}

        # Empty graph (probability 0)
        f_empty = SimpleContinuousGraphon((x, y) -> 0.0, GBMatrix{Bool})
        A_empty = rand(MersenneTwister(555), f_empty, 10)
        @test nnz(A_empty) == 0

        # Full graph (probability 1, except diagonal)
        f_full = SimpleContinuousGraphon((x, y) -> 1.0, GBMatrix{Bool})
        A_full = rand(MersenneTwister(666), f_full, 10)
        @test nnz(A_full) == 10 * 9  # n(n-1) edges
    end

    @testset "Type stability with GBMatrix" begin
        f = SimpleContinuousGraphon((x, y) -> 0.5, GBMatrix{Bool})
        @test Graphons.make_empty_graph(GBMatrix{Bool}, 10) isa GBMatrix{Bool}

        g = DecoratedGraphon((x, y) -> Normal(0, 1), GBMatrix{Float64})
        @test Graphons.make_empty_graph(GBMatrix{Float64}, 10) isa GBMatrix{Float64}
    end

    @testset "Conversion and comparison" begin
        rng = MersenneTwister(777)
        f = SimpleContinuousGraphon((x, y) -> 0.5)

        # Sample with regular BitMatrix
        A_bit = rand(rng, f, 10)

        # Sample with GBMatrix
        rng2 = MersenneTwister(777)
        f_gb = SimpleContinuousGraphon((x, y) -> 0.5, GBMatrix{Bool})
        A_gb = rand(rng2, f_gb, 10)

        # They should produce the same graph structure
        @test A_bit == Matrix(A_gb)
    end
end
