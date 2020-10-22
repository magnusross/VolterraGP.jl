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

# @btime negloglikelihood(gp)
# @btime gradient(negloglikelihood, gp)
# @btime gradient(Flux.params(gp.dpars.σ, gp.dpars.G, gp.dpars.u)) do
#     negloglikelihood(gp)
# end

# a = Zygote.forward_jacobian(negloglikelihood, gp)
function sig_like(sig, gp)
    gp.dpars.σ = sig
    negloglikelihood(gp)
end
a = Zygote.forward_jacobian(x -> sig_like(x, gp), ones(3))
f = x -> x' * x
b = Zygote.forward_jacobian(f, ones(5))[2]