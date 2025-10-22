using Graphons
using Test
using Random
using Distributions
using Clustering

@testset "ClusteringExt" begin
    @testset "SSM from SBM - Basic" begin
        # Create a simple SBM with clear structure
        θ = [0.9 0.3 0.1;
             0.3 0.8 0.3;
             0.1 0.3 0.9]
        sizes = [0.33, 0.34, 0.33]
        sbm = SBM(θ, sizes)

        # Convert to SSM with 3 shapes (should recover unique values)
        ssm = Graphons.SSM(sbm, 3)

        @test ssm isa Graphons.SSM
        @test length(ssm.θ) == 3
        @test ssm.size == sizes
        @test size(ssm.block_pair_to_shape) == (3, 3)

        # Verify shape assignments are symmetric
        @test issymmetric(ssm.block_pair_to_shape)
    end

    @testset "SSM from SBM - Shape reduction" begin
        # SBM with repeated probabilities that can be compressed
        θ = [0.9 0.2 0.2;
             0.2 0.9 0.2;
             0.2 0.2 0.9]
        sizes = fill(1 / 3, 3)
        sbm = SBM(θ, sizes)

        # Request fewer shapes than unique block pairs
        ssm = Graphons.SSM(sbm, 2)

        @test ssm isa Graphons.SSM
        @test length(ssm.θ) == 2
        @test size(ssm.block_pair_to_shape) == (3, 3)

        # Verify reasonable compression
        θ_matrix = Graphons.get_theta_matrix(ssm)
        @test size(θ_matrix) == (3, 3)
    end

    @testset "SSM from DecoratedSBM" begin
        # Create DecoratedSBM with Normal distributions
        θ = [Normal(1.0, 0.1) Normal(0.0, 0.1);
             Normal(0.0, 0.1) Normal(1.0, 0.1)]
        sizes = [0.5, 0.5]
        dsbm = DecoratedSBM(θ, sizes)

        # Convert to SSM
        ssm = Graphons.SSM(dsbm, 2)

        @test ssm isa Graphons.SSM
        @test length(ssm.θ) == 2
        @test all(d -> d isa Normal, ssm.θ)
        @test size(ssm.block_pair_to_shape) == (2, 2)
    end

    @testset "SSM from DecoratedSBM - DiscreteNonParametric" begin
        # Test with discrete distributions
        support_vals = [0, 1, 2]
        θ = [DiscreteNonParametric(support_vals, [0.7, 0.2, 0.1]) DiscreteNonParametric(support_vals, [0.1, 0.2, 0.7]);
             DiscreteNonParametric(support_vals, [0.1, 0.2, 0.7]) DiscreteNonParametric(support_vals, [0.3, 0.4, 0.3])]
        sizes = [0.6, 0.4]
        dsbm = DecoratedSBM(θ, sizes)

        ssm = Graphons.SSM(dsbm, 2)

        @test ssm isa Graphons.SSM
        @test length(ssm.θ) == 2
        @test all(d -> d isa DiscreteNonParametric, ssm.θ)
        # Verify support is preserved
        @test all(d -> Distributions.support(d) == support_vals, ssm.θ)
    end

    @testset "Validation" begin
        θ = [0.8 0.2; 0.2 0.7]
        sizes = [0.5, 0.5]
        sbm = SBM(θ, sizes)

        # Invalid number of shapes
        @test_throws Exception Graphons.SSM(sbm, 0)
        @test_throws Exception Graphons.SSM(sbm, -1)

        # Too many shapes (more than unique block pairs)
        # For 2x2 matrix, there are 3 unique upper triangular entries
        @test_throws AssertionError Graphons.SSM(sbm, 10)
    end

    @testset "SSM conversion preserves structure" begin
        # Create SBM and convert to SSM
        θ = [0.9 0.4 0.1;
             0.4 0.8 0.3;
             0.1 0.3 0.7]
        sizes = [0.3, 0.5, 0.2]
        sbm = SBM(θ, sizes)

        # Use enough shapes to capture all unique values
        ssm = Graphons.SSM(sbm, 5)

        # Convert back to SBM
        sbm2 = Graphons.convert_to_sbm(ssm)

        # Verify structure is approximately preserved
        @test sbm2.size == sbm.size
        @test size(sbm2.θ) == size(sbm.θ)

        # Values should be close (kmeans may not match exactly)
        for i in 1:3, j in 1:3
            @test abs(sbm2.θ[i, j] - sbm.θ[i, j]) < 0.5
        end
    end

    @testset "Sampling from SSM via SBM" begin
        rng = MersenneTwister(42)
        θ = [0.9 0.2; 0.2 0.8]
        sizes = [0.5, 0.5]
        sbm = SBM(θ, sizes)

        ssm = Graphons.SSM(sbm, 2)

        # Sample from SSM
        A = rand(rng, ssm, 20)
        @test size(A) == (20, 20)
        @test issymmetric(A)
        @test all(diag(A) .== 0)
    end

    @testset "Matrix type preservation" begin
        θ = [0.8 0.2; 0.2 0.7]
        sizes = [0.5, 0.5]
        sbm = SBM(θ, sizes)

        ssm = Graphons.SSM(sbm, 2)

        # Should preserve matrix type from SBM
        @test Graphons.matrix_type(ssm) == BitMatrix
    end

    @testset "Large SBM compression" begin
        # Create larger SBM
        K = 5
        rng = MersenneTwister(123)
        θ_raw = rand(rng, K, K)
        θ = (θ_raw + θ_raw') / 2  # Make symmetric
        sizes = fill(1 / K, K)
        sbm = SBM(θ, sizes)

        # Compress to fewer shapes
        n_shapes = 3
        ssm = Graphons.SSM(sbm, n_shapes)

        @test ssm isa Graphons.SSM
        @test length(ssm.θ) == n_shapes
        @test size(ssm.block_pair_to_shape) == (K, K)

        # All shape indices should be valid
        @test all(1 .<= ssm.block_pair_to_shape .<= n_shapes)

        # Should be callable
        @test 0 <= ssm(0.3, 0.7) <= 1
    end

    @testset "Edge cases" begin
        # Two blocks, single shape (homogeneous)
        θ = [0.5 0.5; 0.5 0.5]
        sizes = [0.5, 0.5]
        sbm = SBM(θ, sizes)

        ssm = Graphons.SSM(sbm, 1)
        @test length(ssm.θ) == 1
        @test all(ssm.block_pair_to_shape .== 1)

        # Three blocks, two shapes
        θ = [0.8 0.3 0.3;
             0.3 0.8 0.3;
             0.3 0.3 0.8]
        sizes = fill(1 / 3, 3)
        sbm = SBM(θ, sizes)

        ssm = Graphons.SSM(sbm, 2)
        @test length(ssm.θ) == 2
        @test size(ssm.block_pair_to_shape) == (3, 3)
    end
end
