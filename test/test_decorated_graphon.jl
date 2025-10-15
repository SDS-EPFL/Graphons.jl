using Graphons
using Test
using Distributions

@testset "DecoratedGraphon" begin
    @testset "Construction with univariate distribution" begin
        # Graphon returning a distribution
        f = DecoratedGraphon((x, y) -> Bernoulli(x * y))
        @test f(0.5, 0.5) isa Bernoulli

        g = DecoratedGraphon((x, y) -> Normal(x, y))
        @test g(0.5, 0.3) isa Normal
    end

    @testset "Construction with multivariate distribution" begin
        # Multivariate distribution
        h = DecoratedGraphon((x, y) -> MvNormal([x, y], [0.1 0; 0 0.1]))
        @test h(0.5, 0.5) isa MvNormal
    end

    @testset "empirical_graphon from decorated" begin
        f = DecoratedGraphon((x, y) -> Bernoulli(x * y))
        sbm = empirical_graphon(f, 3)

        @test sbm isa DecoratedSBM
        @test size(sbm.θ) == (3, 3)
        @test sum(sbm.size) ≈ 1.0
    end
end
