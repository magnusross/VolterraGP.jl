using VolterraGP
using Test
using BenchmarkTools
using LinearAlgebra
using Flux

P = 1


X = [collect(-1:0.1:1), collect(-1:0.1:1)]
Y = [sin.(X[1]), cos.(X[2])]

Xd = [collect(-1:0.1:1), collect(0:0.2:1.5)]
Yd = [sin.(Xd[1]), cos.(Xd[2])]

t = [collect(-1.5:0.3:1.5), collect(-1.5:0.3:1.5)]

data = Data(X, Y)
datad = Data(Xd, Yd)

dpars1 = DiffableParameters([0.1, 0.1], ones((2, sum(1:1), P)), [0.1])
gp1 = GaussianProcess(threeEQs, 2, 1, P, data, dpars1)


dpars2 = DiffableParameters([0.1, 0.1], ones((2, sum(1:2), P)), [0.1])
gp2 = GaussianProcess(threeEQs, 2, 2, P, data, dpars2)

dparsrand = DiffableParameters([0.3, 0.4], 0.1 .+ rand(Float64, (2, sum(1:2), P)), [0.1])
gprand = GaussianProcess(threeEQs, 2, 2, P, data, dparsrand)

gpd = GaussianProcess(threeEQs, 2, 2, P, datad, dpars2)

dparssca = DiffableParameters([0.1, 0.1], 0.5ones((2, sum(1:2), 2)), [0.1])
gpscaled = GaussianProcess(scaledEQs, 2, 2, P, data, dparssca)

test_psd = K -> minimum(eigvals(K)) > -sqrt(eps())
test_herm = K -> maximum(K' - K) < sqrt(eps())

# gp3 


@testset "VolterraGP.jl" begin
    @test negloglikelihood(gp1) ≈ 475.6091937696483
    @test negloglikelihood(gp2) ≈ 476.28491875918456
    @test negloglikelihood(gpd) ≈ 48.92347237044826
    @test negloglikelihood(gpscaled) ≈ 477.9933646808806
end

@testset "volterra.jl" begin
    # when C=1 running thru volterra should be same as base 
    @test VolterraGP.kernel(1., 2., 1, 2, gp1) ≈ threeEQs(1., 2., gp1.dpars.G[1, 1, :], gp1.dpars.G[2, 1, :], gp1.dpars.u)
    @test VolterraGP.kernel(1., 2., 1, 1, gp1) ≈ threeEQs(1., 2., gp1.dpars.G[1, 1, :], gp1.dpars.G[1, 1, :], gp1.dpars.u)

    # test for symmerty in kernel arguments 
    @test VolterraGP.kernel(1., 2., 1, 2, gprand) ≈ VolterraGP.kernel(1., 2., 2, 1, gprand)
    @test VolterraGP.kernel(1., 2., 1, 1, gprand) ≈ VolterraGP.kernel(2., 1., 1, 1, gprand)
    
    # examples from paper
    @test VolterraGP.kan_rv_prod(ones(Float64, (4, 4))) ≈ 3.
    @test VolterraGP.kan_rv_prod(ones(Float64, (2, 2))) ≈ 1.
    
    phi_E = VolterraGP.get_phi_E(0.2, 2, 1, gprand)
    @test test_psd(phi_E)
    @test test_herm(phi_E)

    phi_K = VolterraGP.get_phi_cov(0.1, 0.2, 2, 2, 2, 2, gprand)
    @test test_psd(phi_K)
    @test test_herm(phi_K)

end

@testset "gp.jl" begin  
    for gp in (gp1, gp2, gpd, gpscaled)

        K = VolterraGP.fill_K(t, t, gp)
        μp, Kp = posterior(t, gp)

        @test test_psd(K)
        @test test_herm(K)
        @test test_psd(Kp)
        @test test_herm(Kp)
    end

end

@testset "gradients" begin
    # test gradient works 
    g1 = gradient(negloglikelihood, gp1) 
    @test g1 == g1
    g2 = gradient(negloglikelihood, gp2) 
    @test g2 == g2
    gr = gradient(negloglikelihood, gprand) 
    @test gr == gr
    gd = gradient(negloglikelihood, gpd) 
    @test gd == gd
    gr = gradient(negloglikelihood, gpscaled)
    @test gr == gr

    # test forward vs reverse gradients 
    g_fwd = VolterraGP.fwd_grad_like(gpscaled)
    g_bwd = Flux.gradient(Flux.params(gpscaled.dpars.σ, gpscaled.dpars.G, gpscaled.dpars.u)) do
        negloglikelihood(gpscaled)
    end

    
    @test all(g_bwd[gpscaled.dpars.σ] .≈ g_fwd[1])
    @test all(g_bwd[gpscaled.dpars.G] .≈ g_fwd[2])
    @test all(g_bwd[gpscaled.dpars.u] .≈ g_fwd[3])


end

