using Graphons
using Test
using Random

@testset "SBM" begin
    @testset "Construction" begin
        θ = [0.8 0.1; 0.1 0.9]
        sizes = [0.5, 0.5]
        sbm = SBM(θ, sizes)

        @test sbm isa SBM
        @test sbm.θ == θ
        @test sbm.size == sizes
        @test sbm.cumsize ≈ [0.5, 1.0]

        # Test validation
        @test_throws Exception SBM([1.5 0.1; 0.1 0.9], sizes)  # Invalid probability
        @test_throws Exception SBM(θ, [0.5, 0.6])  # Sizes don't sum to 1
        @test_throws Exception SBM(θ, [-0.1, 1.1])  # Negative size
    end

    @testset "Evaluation" begin
        θ = [0.8 0.2; 0.2 0.6]
        sizes = [0.4, 0.6]
        sbm = SBM(θ, sizes)

        # Test within block 1
        @test sbm(0.1, 0.2) == 0.8
        @test sbm(0.3, 0.35) == 0.8

        # Test within block 2
        @test sbm(0.5, 0.7) == 0.6
        @test sbm(0.9, 0.95) == 0.6

        # Test cross-block
        @test sbm(0.1, 0.7) == 0.2
        @test sbm(0.7, 0.1) == 0.2
    end

    @testset "Edge cases" begin
        # Single block
        θ = reshape([0.5], 1, 1)
        sizes = [1.0]
        sbm = SBM(θ, sizes)

        A = rand(MersenneTwister(369), sbm, 5)
        @test size(A) == (5, 5)
        @test issymmetric(A)

        # Many blocks
        k = 5
        θ = rand(MersenneTwister(147), k, k)
        θ = (θ + θ') / 2  # Make symmetric
        sizes = fill(1 / k, k)
        sbm = SBM(θ, sizes)

        A = rand(MersenneTwister(258), sbm, 20)
        @test size(A) == (20, 20)
        @test issymmetric(A)
    end

    @testset "Type stability" begin
        sbm = SBM([0.8 0.2; 0.2 0.7], [0.5, 0.5])
        @inferred sbm(0.3, 0.7)
    end
end
