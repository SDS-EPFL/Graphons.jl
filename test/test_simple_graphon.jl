using Graphons
using Test

@testset "SimpleContinuousGraphon" begin
    @testset "Construction and evaluation" begin
        # Simple constant graphon
        f = SimpleContinuousGraphon((x, y) -> 0.5)
        @test f(0.1, 0.2) == 0.5
        @test f(0.8, 0.9) == 0.5

        # Position-dependent graphon
        g = SimpleContinuousGraphon((x, y) -> x * y)
        @test g(0.5, 0.5) ≈ 0.25
        @test g(0.0, 1.0) ≈ 0.0

        # Test with custom matrix type
        h = SimpleContinuousGraphon((x, y) -> 0.3, SparseMatrixCSC{Bool,Int})
        @test h(0.1, 0.2) == 0.3
    end

    @testset "empirical_graphon from continuous" begin
        f = SimpleContinuousGraphon((x, y) -> x * y)
        sbm = empirical_graphon(f, 3)

        @test sbm isa SBM
        @test size(sbm.θ) == (3, 3)
        @test sum(sbm.size) ≈ 1.0
        @test all(sbm.θ .>= 0) && all(sbm.θ .<= 1)
    end

    @testset "Type stability" begin
        f = SimpleContinuousGraphon((x, y) -> 0.5)
        @inferred f(0.5, 0.5)
    end
end
