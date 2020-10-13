using VolterraGP
using Test
using BenchmarkTools


P = 1


X = collect(-1:0.1:1)
Y = [sin.(X), cos.(X)]

data = Data(X, Y)


dpars = DiffableParameters([0.1, 0.1], ones(Float64, (2, sum(1:1), P)), [0.1])
gp = GaussianProcess(threeEQs, 2, 1, P, data, dpars)


@testset "VolterraGP.jl" begin
    @test negloglikelihood(gp) ≈ 475.6091937696483
end

@testset "volterra.jl" begin
    # when C=1 running thru volterra should be same as base 
    @test VolterraGP.kernel(1., 2., 1, 2, gp) ≈ threeEQs(1., 2., gp.dpars.G[1, 1, 1], gp.dpars.G[2, 1, 1], gp.dpars.u)
    @test VolterraGP.kernel(1., 2., 1, 1, gp) ≈ threeEQs(1., 2., gp.dpars.G[1, 1, 1], gp.dpars.G[1, 1, 1], gp.dpars.u)

    @test VolterraGP.kan_rv_prod(ones(4, 4)) ≈ 3.
    @test VolterraGP.kan_rv_prod(ones(2, 2)) ≈ 1.
end

@testset "speed" begin

    # like_allocs = (@timed negloglikelihood(gp))[3]
    # grad_allocs = (@timed gradient(Flux.params(gp.dpars.σ, gp.dpars.G, gp.dpars.u)) do
    #     negloglikelihood(gp)
    # end)[3]
    # best_like, best_grad = readlines("test/best_allocs.txt")
    
    # @test like_allocs <= parse(Int64, best_like)*1.05
    # @test grad_allocs <= parse(Int64, best_grad)*1.05
    

 
end
