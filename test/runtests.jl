using VolterraGP
using Test
using BenchmarkTools


P = 1

dpars1 = DiffableParameters([0.1], ones(Float64, (1, sum(1:2), P)), [0.1])

X = collect(-1:0.1:1)
Y = [sin.(X)]
data = Data(X, Y)

gp1 = GaussianProcess(threeEQs, 1, 2, P, data, dpars1)

dpars2 = DiffableParameters([0.1], ones(Float64, (2, sum(1:1), P)), [0.1])
gp2 = GaussianProcess(threeEQs, 2, 1, P, data, dpars2)

@testset "VolterraGP.jl" begin
    # @test abs(negloglikelihood(gp) - (-11.65748958085625)) <=  eps()
end

@testset "volterra.jl" begin
    # when C=1 running thru volterra should be same as base 
    @test VolterraGP.kernel(1., 2., 1, 2, gp2) == threeEQs(1., 2., gp2.dpars.G[1, 1, 1], gp2.dpars.G[2, 1, 1], gp2.dpars.u)
    @test VolterraGP.kernel(1., 2., 1, 1, gp2) == threeEQs(1., 2., gp2.dpars.G[1, 1, 1], gp2.dpars.G[1, 1, 1], gp2.dpars.u)

    @test VolterraGP.kan_rv_prod(ones(4, 4)) == 3.
    @test VolterraGP.kan_rv_prod(ones(2, 2)) == 1.
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
