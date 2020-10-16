using VolterraGP
using Test
using BenchmarkTools
using LinearAlgebra
using Flux

P = 1


X = collect(-1:0.1:1)
Y = [sin.(X), cos.(X)]

t = collect(-1.5:0.3:1.5)

data = Data(X, Y)

dpars1 = DiffableParameters([0.1, 0.1], ones(Float64, (2, sum(1:1), P)), [0.1])
gp1 = GaussianProcess(threeEQs, 2, 1, P, data, dpars1)


dpars2 = DiffableParameters([0.1, 0.1], ones(Float64, (2, sum(1:2), P)), [0.1])
gp2 = GaussianProcess(threeEQs, 2, 2, P, data, dpars2)

dparsrand = DiffableParameters([0.3, 0.4], 0.1 .+ rand(Float64, (2, sum(1:2), P)), [0.1])
gprand = GaussianProcess(threeEQs, 2, 2, P, data, dparsrand)

test_psd = K -> minimum(eigvals(K)) > -sqrt(eps())
test_herm = K -> maximum(K' - K) < sqrt(eps())



@testset "VolterraGP.jl" begin
    @test negloglikelihood(gp1) ≈ 475.6091937696483
    @test negloglikelihood(gp2) ≈ 476.28491875918456
end

@testset "volterra.jl" begin
    # when C=1 running thru volterra should be same as base 
    @test VolterraGP.kernel(1., 2., 1, 2, gp1) ≈ threeEQs(1., 2., gp1.dpars.G[1, 1, 1], gp1.dpars.G[2, 1, 1], gp1.dpars.u)
    @test VolterraGP.kernel(1., 2., 1, 1, gp1) ≈ threeEQs(1., 2., gp1.dpars.G[1, 1, 1], gp1.dpars.G[1, 1, 1], gp1.dpars.u)

    # test for symmerty in kernel arguments 
    @test VolterraGP.kernel(1., 2., 1, 2, gprand) ≈ VolterraGP.kernel(1., 2., 2, 1, gprand)
    @test VolterraGP.kernel(1., 2., 1, 1, gprand) ≈ VolterraGP.kernel(2., 1., 1, 1, gprand)
    
    # examples from paper
    @test VolterraGP.kan_rv_prod(ones(4, 4)) ≈ 3.
    @test VolterraGP.kan_rv_prod(ones(2, 2)) ≈ 1.
    
    phi_E = VolterraGP.get_phi_E(0.2, 2, 1, gprand)
    @test test_psd(phi_E)
    @test test_herm(phi_E)

    phi_K = VolterraGP.get_phi_cov(0.1, 0.2, 2, 2, 2, 2, gprand)
    @test test_psd(phi_K)
    @test test_herm(phi_K)

end

@testset "gp.jl" begin

    K1 = VolterraGP.fill_K(t, t, gp1)
    @test test_psd(K1)
    @test test_herm(K1)

    K2 = VolterraGP.fill_K(t, t, gp2)
    @test test_psd(K2)
    @test test_herm(K2)

    Kr = VolterraGP.fill_K(t, t, gprand)
    @test test_psd(Kr)
    @test test_herm(Kr)

    μp1, Kp1 = posterior(t, gp1)
    @test test_psd(Kp1)
    @test test_herm(Kp1)

    μp2, Kp2 = posterior(t, gp2)
    @test test_psd(Kp2)
    @test test_herm(Kp2)

    μpr, Kpr = posterior(t, gprand)
    @test test_psd(Kpr)
    @test test_herm(Kpr)

end

@testset "gradients" begin
    # test gradient works 
    g1 = gradient(negloglikelihood, gp1) 
    @test g1 == g1
    g2 = gradient(negloglikelihood, gp2) 
    @test g2 == g2
    gr = gradient(negloglikelihood, gprand) 
    @test gr == gr

    # # test custom adjoint 
    # function kan_rv_prod_test(phi::Array{Float64,2})::Float64
    #     st = size(phi)[1]
    #     # mapreduce(v -> VolterraGP.kan_rv_prod_inner(phi, v), +, Iterators.product(fill(0:1, st)...)) / factorial(st ÷ 2)
    #     1.
    # end


    # phi = ones(4, 4)

    # g_custom = gradient(VolterraGP.kan_rv_prod, phi)
    # g_test = gradient(kan_rv_prod_test,  phi)
    # t = gradient(VolterraGP.full_E, 0.1, 1, gp1)

    # print(g_custom)

    

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
