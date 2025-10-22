using Graphons
using Test
using SparseArrays
using Distributions

@testset "Utils" begin
    @testset "make_empty_graph" begin
        # Test Matrix creation
        A = Graphons.make_empty_graph(Matrix{Float64}, 5)
        @test size(A) == (5, 5)
        @test eltype(A) == Float64
        @test all(A .== 0)

        # Test BitMatrix creation
        B = Graphons.make_empty_graph(BitMatrix, 4)
        @test size(B) == (4, 4)
        @test eltype(B) == Bool
        @test all(.!B)

        # Test sparse matrix creation
        C = Graphons.make_empty_graph(SparseMatrixCSC{Float64, Int}, 3)
        @test size(C) == (3, 3)
        @test eltype(C) == Float64
        @test nnz(C) == 0
    end

    @testset "clear_graph!" begin
        # Test clearing dense matrix
        A = ones(3, 3)
        Graphons.clear_graph!(A)
        @test all(A .== 0)

        # Test clearing sparse matrix
        B = sparse([1, 2], [2, 3], [1.0, 2.0], 3, 3)
        Graphons.clear_graph!(B)
        @test nnz(B) == 0
    end

    @testset "_convert_latent_to_block" begin
        sizes = [0.3, 0.3, 0.4]
        sbm = SBM([0.8 0.1 0.2; 0.1 0.9 0.1; 0.2 0.1 0.7], sizes)

        @test Graphons._convert_latent_to_block(sbm, 0.15) == 1
        @test Graphons._convert_latent_to_block(sbm, 0.45) == 2
        @test Graphons._convert_latent_to_block(sbm, 0.8) == 3

        @test Graphons._convert_latent_to_block(sbm, 0.15, 0.45) == (1, 2)
        @test Graphons._convert_latent_to_block(sbm, 0.5, 0.85) == (2, 3)
    end

    @testset "_extract_param" begin
        @testset "Univariate continuous distributions" begin
            # Normal distribution (μ, σ)
            d = Normal(2.5, 1.5)
            result = Graphons._extract_param(d, :)
            @test result isa AbstractVector{<:AbstractFloat}
            @test result == [2.5, 1.5]

            # Exponential distribution (θ)
            d = Exponential(3.0)
            result = Graphons._extract_param(d, :)
            @test result isa AbstractVector{<:AbstractFloat}
            @test length(result) == 1
            @test result[1] == 3.0

            # Beta distribution (α, β)
            d = Beta(2.0, 5.0)
            result = Graphons._extract_param(d, :)
            @test result isa AbstractVector{<:AbstractFloat}
            @test result == [2.0, 5.0]

            # Gamma distribution (α, θ)
            d = Gamma(3.0, 2.0)
            result = Graphons._extract_param(d, :)
            @test result isa AbstractVector{<:AbstractFloat}
            @test length(result) == 2

            # Uniform distribution (a, b)
            d = Uniform(0.0, 1.0)
            result = Graphons._extract_param(d, :)
            @test result isa AbstractVector{<:AbstractFloat}
            @test result == [0.0, 1.0]
        end

        @testset "Univariate discrete distributions" begin
            # Bernoulli distribution (p)
            d = Bernoulli(0.7)
            result = Graphons._extract_param(d, :)
            @test result isa AbstractVector{<:AbstractFloat}
            @test length(result) == 1
            @test result[1] ≈ 0.7

            # Binomial distribution (n, p)
            d = Binomial(10, 0.3)
            result = Graphons._extract_param(d, :)
            @test result isa AbstractVector{<:AbstractFloat}
            @test length(result) == 2
            @test result[1] == 10
            @test result[2] == 0.3
            @test Graphons._extract_param(d, 1) == 10
            @test Graphons._extract_param(d, 2) == 0.3

            # Poisson distribution (λ)
            d = Poisson(4.5)
            result = Graphons._extract_param(d, :)
            @test result isa AbstractVector{<:AbstractFloat}
            @test length(result) == 1
            @test result[1] ≈ 4.5

            # Geometric distribution (p)
            d = Geometric(0.25)
            result = Graphons._extract_param(d, :)
            @test result isa AbstractVector{<:AbstractFloat}
            @test length(result) == 1
        end

        @testset "DiscreteNonParametric distribution" begin
            # DiscreteNonParametric has special handling
            support = [1, 2, 3, 4]
            probs = [0.1, 0.2, 0.3, 0.4]
            d = DiscreteNonParametric(support, probs)
            result = Graphons._extract_param(d, :)
            @test result isa AbstractVector{<:AbstractFloat}
            @test result == probs
        end

        @testset "Single parameter extraction" begin
            # Test extracting individual parameters
            d = Normal(2.5, 1.5)
            @test Graphons._extract_param(d, 1) == 2.5
            @test Graphons._extract_param(d, 2) == 1.5

            d = Bernoulli(0.8)
            @test Graphons._extract_param(d, 1) ≈ 0.8

            # DiscreteNonParametric single parameter
            support = [1, 2, 3]
            probs = [0.2, 0.3, 0.5]
            d = DiscreteNonParametric(support, probs)
            @test Graphons._extract_param(d, 1) == 0.2
            @test Graphons._extract_param(d, 2) == 0.3
            @test Graphons._extract_param(d, 3) == 0.5
        end

        @testset "Multivariate distributions" begin
            # MvNormal distribution
            μ = [1.0, 2.0]
            Σ = [1.0 0.5; 0.5 2.0]
            d = MvNormal(μ, Σ)
            result = Graphons._extract_param(d, :)
            @test result isa AbstractVector{<:AbstractFloat}
            # For MvNormal, params returns (μ, Σ)
            # The result should flatten both mean and covariance
            @test length(result) >= 2  # At least mean parameters
            # Check that mean values are present in the result
            @test result[1] == 1.0
            @test result[2] == 2.0
        end
    end
end
