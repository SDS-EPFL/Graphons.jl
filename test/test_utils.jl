using Graphons
using Test
using SparseArrays

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
        C = Graphons.make_empty_graph(SparseMatrixCSC{Float64,Int}, 3)
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
        cumsizes = cumsum(sizes)
        sbm = SBM([0.8 0.1 0.2; 0.1 0.9 0.1; 0.2 0.1 0.7], sizes)

        @test Graphons._convert_latent_to_block(sbm, 0.15) == 1
        @test Graphons._convert_latent_to_block(sbm, 0.45) == 2
        @test Graphons._convert_latent_to_block(sbm, 0.8) == 3
    end
end
