using Revise
using VolterraGP
using BenchmarkTools
using Flux
using Zygote 

D = 3
C = 2
P = 2

train, test = generate_toy_data(10)
gp = GaussianProcess(scaledEQs, D, C, P, train) 

# println(VolterraGP.map_prod(v -> VolterraGP.kan_rv_prod_inner_E(0.1, 2, 1, gp, collect(v)), 2))
# println( gradient(VolterraGP.map_prod, v -> VolterraGP.kan_rv_prod_inner_E(0.1, 2, 1, gp, collect(v)), 2))
# println(VolterraGP.full_E(0.1, 2, gp))
# a = gradient(VolterraGP.full_E, 0.1, 2, gp)
# println(VolterraGP.full_cov(0.1, 0.2, 2, 2, gp))
# a = gradient(VolterraGP.full_cov, 0.1, 0.2, 2, 2, gp)
# println(VolterraGP.kan_rv_prod_inner_E, 0.1, 2, 1, gp, (1, 1)))
# println(sum(VolterraGP.kan_rv_prod_E(0.1, 2, 1, gp)))
# gradient(x -> VolterraGP.kan_rv_prod_E(0.1, 2, 1, x), gp) 

# scaledEQs(0.1, 0.1, [0.1, 0.1], [0.1, 0.1], [0.1])

# function kan_rv_prod_inner_E_adj(t, c, d, gp)
#     g = v -> gradient(x -> VolterraGP.kan_rv_prod_inner_E(t, c, d, x, collect(v)), gp)
#     map(g, Iterators.product(fill(0:1, c)...)) # / factorial(c ÷ 2)
# end 

# a =  kan_rv_prod_inner_E_adj(0.1, 2, 1, gp)
# print(VolterraGP.kan_rv_prod_inner_E(0.1, 2, 3, gp, (1, 1)))
# nkE = gradient(VolterraGP.kan_rv_prod_inner_E, 0.1, 2, 3, gp, (1, 1))
# println(kE - nkE)


# nkk = gradient(VolterraGP.kan_rv_prod_cov, 0.1, 0.1,  2, 2, 3, 3, gp)


@btime VolterraGP.full_E(0.1, 2, gp)
@btime VolterraGP.full_cov(0.1, 0.2, 2, 2, gp) # 473.389


# a = Zygote.forward_jacobian(neglopglikelihood, gp)
# function sig_like(sig, gp)
#     dp = DiffableParameters(sig, gp.dpars.G, gp.dpars.u)
#     gp.dpars = dp
#     negloglikelihood(gp)
# end

# # @btime gradient(x -> sig_like(x, gp), ones(3))
# @btime gradient(negloglikelihood, gp)
# # f = (x, y) -> x' * x * y' * y
# b = Zygote.forward_jacobian(f, ones(5), ones(5))