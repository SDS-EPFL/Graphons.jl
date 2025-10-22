using Graphons
using Test
using Random
using Distributions
using LinearAlgebra

@testset "SSM (Stochastic Shape Model)" begin
    @testset "Construction - Simple SSM" begin
        # 2-block model with 2 shapes
        θ = [0.8, 0.2]
        block_pair_to_shape = [1 2; 2 1]
        sizes = [0.5, 0.5]
        cumsizes = cumsum(sizes)

        ssm = Graphons.SSM(θ, block_pair_to_shape, sizes, cumsizes)

        @test ssm isa Graphons.SSM
        @test ssm.θ == θ
        @test ssm.block_pair_to_shape == block_pair_to_shape
        @test ssm.size == sizes
        @test ssm.cumsize ≈ cumsizes
        @test Graphons.edge_type(ssm) == Bool
        @test Graphons.matrix_type(ssm) == BitMatrix
    end

    @testset "Construction - Decorated SSM" begin
        # SSM with distributions
        θ = [Normal(1.0, 0.1), Normal(0.0, 0.1)]
        block_pair_to_shape = [1 2; 2 1]
        sizes = [0.4, 0.6]
        cumsizes = cumsum(sizes)

        ssm = Graphons.SSM(θ, block_pair_to_shape, sizes, cumsizes)

        @test ssm isa Graphons.SSM
        @test ssm.θ == θ
        @test Graphons.edge_type(ssm) == Float64
    end

    @testset "Construction validation" begin
        θ = [0.8, 0.2]
        sizes = [0.5, 0.5]
        cumsizes = cumsum(sizes)

        # Mismatched block_pair_to_shape dimensions
        @test_throws Exception Graphons.SSM(
            θ, [1 2; 2 1; 3 3], sizes, cumsizes)

        # Mismatched sizes and block_pair_to_shape
        @test_throws Exception Graphons.SSM(
            θ, [1 2; 2 1], [0.3, 0.3, 0.4], cumsum([0.3, 0.3, 0.4]))

        # Zero or negative sizes
        @test_throws Exception Graphons.SSM(
            θ, [1 2; 2 1], [0.5, 0.0], [0.5, 0.5])

        # Sizes don't sum to 1
        @test_throws Exception Graphons.SSM(
            θ, [1 2; 2 1], [0.5, 0.6], [0.5, 1.1])
    end

    @testset "Evaluation" begin
        θ = [0.9, 0.3, 0.1]
        block_pair_to_shape = [1 2 3; 2 1 2; 3 2 1]
        sizes = [0.3, 0.4, 0.3]
        cumsizes = cumsum(sizes)

        ssm = Graphons.SSM(θ, block_pair_to_shape, sizes, cumsizes)

        # Within block 1
        @test ssm(0.1, 0.2) == 0.9

        # Within block 2
        @test ssm(0.5, 0.6) == 0.9

        # Within block 3
        @test ssm(0.8, 0.9) == 0.9

        # Cross-block interactions
        @test ssm(0.1, 0.5) == 0.3  # block 1 -> 2, shape 2
        @test ssm(0.1, 0.8) == 0.1  # block 1 -> 3, shape 3
        @test ssm(0.5, 0.8) == 0.3  # block 2 -> 3, shape 2

        # Symmetry
        @test ssm(0.5, 0.1) == ssm(0.1, 0.5)
    end

    @testset "get_theta_matrix" begin
        θ = [0.8, 0.2]
        block_pair_to_shape = [1 2; 2 1]
        sizes = [0.5, 0.5]
        ssm = Graphons.SSM(θ, block_pair_to_shape, sizes, cumsum(sizes))

        θ_matrix = Graphons.get_theta_matrix(ssm)

        @test size(θ_matrix) == (2, 2)
        @test θ_matrix[1, 1] == 0.8
        @test θ_matrix[1, 2] == 0.2
        @test θ_matrix[2, 1] == 0.2
        @test θ_matrix[2, 2] == 0.8
    end

    @testset "convert_to_sbm - Simple" begin
        θ = [0.8, 0.3, 0.1]
        block_pair_to_shape = [1 2 3; 2 1 2; 3 2 1]
        sizes = [0.3, 0.4, 0.3]
        ssm = Graphons.SSM(θ, block_pair_to_shape, sizes, cumsum(sizes))

        sbm = Graphons.convert_to_sbm(ssm)

        @test sbm isa SBM
        @test size(sbm.θ) == (3, 3)
        @test sbm.size == sizes
        @test sbm.θ[1, 1] == 0.8
        @test sbm.θ[1, 2] == 0.3
        @test sbm.θ[2, 3] == 0.3

        # Test that SSM and SBM give same values
        @test sbm(0.1, 0.2) == ssm(0.1, 0.2)
        @test sbm(0.5, 0.8) == ssm(0.5, 0.8)
    end

    @testset "convert_to_sbm - Decorated" begin
        θ = [Normal(1.0, 0.1), Normal(0.0, 0.1), Normal(-1.0, 0.1)]
        block_pair_to_shape = [1 2 3; 2 1 2; 3 2 1]
        sizes = [0.3, 0.4, 0.3]
        ssm = Graphons.SSM(θ, block_pair_to_shape, sizes, cumsum(sizes))

        dsbm = Graphons.convert_to_sbm(ssm)

        @test dsbm isa DecoratedSBM
        @test size(dsbm.θ) == (3, 3)
        @test dsbm.size == sizes
        @test dsbm.θ[1, 1] isa Normal
        @test dsbm.θ[1, 1].μ == 1.0
        @test dsbm.θ[1, 2].μ == 0.0
        @test dsbm.θ[2, 3].μ == 0.0

        # Test that SSM and DecoratedSBM give same distributions
        @test dsbm(0.1, 0.2).μ == ssm(0.1, 0.2).μ
    end

    @testset "Sampling" begin
        rng = MersenneTwister(42)
        θ = [0.9, 0.1]
        block_pair_to_shape = [1 2; 2 1]
        sizes = [0.5, 0.5]
        ssm = Graphons.SSM(θ, block_pair_to_shape, sizes, cumsum(sizes))

        # Test rand
        A = rand(rng, ssm, 20)
        @test size(A) == (20, 20)
        @test issymmetric(A)
        @test all(diag(A) .== 0)  # No self-loops

        # Test sample_graph with fixed latents
        ξs = range(0, 1, length = 10)
        A2 = Graphons.sample_graph(rng, ssm, ξs)
        @test size(A2) == (10, 10)
        @test issymmetric(A2)
    end

    @testset "Multiple shapes" begin
        # Test with many shapes
        n_shapes = 5
        θ = rand(MersenneTwister(123), n_shapes)
        K = 4
        block_pair_to_shape = rand(MersenneTwister(456), 1:n_shapes, K, K)
        block_pair_to_shape = (block_pair_to_shape + block_pair_to_shape') .÷ 2
        sizes = fill(1 / K, K)

        ssm = Graphons.SSM(θ, block_pair_to_shape, sizes, cumsum(sizes))

        @test ssm isa Graphons.SSM
        @test length(unique(block_pair_to_shape)) <= n_shapes

        # Verify it can be evaluated and sampled
        @test 0 <= ssm(0.2, 0.7) <= 1
        A = rand(MersenneTwister(789), ssm, 15)
        @test size(A) == (15, 15)
    end
end
