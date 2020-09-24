using VolterraGP: kan_rv_prod, DiffableParameters 
using Test
using BenchmarkTools

D = 1
C = 2 
P = 1

dpars = DiffableParameters([0.1], ones(Float64, (D, sum(1:C), P)), [0.1])

X = collect(-1:0.1:1)
Y = sin.(X)
data = Data(X, Y)

gp = GaussianProcess(threeEQs, D, C, P, data, dpars)


@testset "VolterraGP.jl" begin
    @test abs(negloglikelihood(gp) - (-11.65748958085625)) <=  eps()
end

@testset "volterra.jl" begin
    @test kan_rv_prod(ones(4, 4), ones(Int64, 4)) == 3.
    @test kan_rv_prod(ones(2, 2), ones(Int64, 2) .* 3) == 15.
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
