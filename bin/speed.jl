using Revise
using VolterraGP
using BenchmarkTools
using Flux
using Zygote
using LinearAlgebra 

D = 3
C = 2
P = 2

train, test = generate_toy_data(10)
gp = GaussianProcess(scaledEQs, D, C, P, train) 

# @btime negloglikelihood(gp)
# @btime gradient(negloglikelihood, gp)

# @btime VolterraGP.kernel(0.1, 0.2, 1, 1, gp)
# _, back = Zygote.pullback(VolterraGP.kernel, 0.1, 0.2, 1, 1, gp)
# @btime back(I)

# @btime VolterraGP.full_cov(0.1, 0.2, 1, 1, gp)
# _, back = Zygote.pullback(VolterraGP.full_cov, 0.1, 0.2, 1, 1, gp)
# @btime back(I)

# @btime sum(VolterraGP.get_phi_cov(0.1, 0.2, 2, 2, 1, 1, gp))
# @btime gradient(pars -> sum(VolterraGP.get_phi_cov(pars...)), (0.1, 0.2, 2, 2, 1, 1, gp))

# @btime sum(VolterraGP.fill_phi(gp.base_kernel, ones(4), ones(4, 2), [1., 1.]))
# @btime gradient(pars -> sum(VolterraGP.fill_phi(pars...)), (gp.base_kernel, ones(4), ones(4, 2), [0.1]))


# phi = VolterraGP.get_phi_cov(0.1, 0.2, 2, 2, 1, 1, gp)

# @btime VolterraGP.kan_rv_prod(phi)
# @btime gradient(VolterraGP.kan_rv_prod, phi)

# @btime VolterraGP.scaledEQs(0.1, 0.2, ones(2), ones(2), [1.])
# @btime gradient(VolterraGP.scaledEQs, 0.1, 0.2, ones(2), ones(2), [1.])
# println(VolterraGP.scaledEQs_adj(2., 0.1, 0.2, ones(2), ones(2), [1.]))


# # a = Zygote.forward_jacobian(negloglikelihood, gp)
# function sig_like(sig, gp)
#     gp.dpars.σ = sig
#     negloglikelihood(gp)
# end
# a = Zygote.forward_jacobian(x -> sig_like(x, gp), ones(3))
# f = x -> x' * x