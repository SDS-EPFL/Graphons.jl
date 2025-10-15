using Graphons
using Test
using Distributions

@testset "DecoratedSBM" begin
    @testset "Construction" begin
        θ = [Bernoulli(0.8) Bernoulli(0.1); Bernoulli(0.1) Bernoulli(0.9)]
        sizes = [0.5, 0.5]
        dsbm = DecoratedSBM(θ, sizes)

        @test dsbm isa DecoratedSBM
        @test size(dsbm.θ) == (2, 2)
        @test dsbm.size == sizes
        @test dsbm.cumsize ≈ [0.5, 1.0]

        # Test validation
        @test_throws Exception DecoratedSBM(θ, [0.5, 0.6])  # Sizes don't sum to 1
    end

    @testset "Evaluation" begin
        θ = [Bernoulli(0.8) Bernoulli(0.2); Bernoulli(0.2) Bernoulli(0.6)]
        sizes = [0.4, 0.6]
        dsbm = DecoratedSBM(θ, sizes)

        # Test within block 1
        @test dsbm(0.1, 0.2) isa Bernoulli
        @test dsbm(0.1, 0.2).p == 0.8

        # Test within block 2
        @test dsbm(0.5, 0.7) isa Bernoulli
        @test dsbm(0.5, 0.7).p == 0.6

        # Test cross-block
        @test dsbm(0.1, 0.7).p == 0.2
    end
end
