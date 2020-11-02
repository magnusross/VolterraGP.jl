using VolterraGP
using ForwardDiff
using Zygote
using BenchmarkTools

function fwd_grad_like(gp)
    dσ = ForwardDiff.gradient(x -> negloglikelihood(x, gp.dpars.G, gp.dpars.u, gp), gp.dpars.σ)
    dG = ForwardDiff.gradient(x -> negloglikelihood(gp.dpars.σ, x, gp.dpars.u, gp), gp.dpars.G)
    du = ForwardDiff.gradient(x -> negloglikelihood(gp.dpars.σ, gp.dpars.G, x, gp), gp.dpars.u)
    (dσ, dG, du)
end 

D = 3
C = 3
P = 2

train, test = generate_toy_data(20)
gp = GaussianProcess(scaledEQs, D, C, P, train) 
# fwd_grad_like(gp)
# @btime Zygote.gradient(negloglikelihood, gp)
@btime fwd_grad_like(gp)

# 3.488 s (26768965 allocations: 1.93 GiB)
# 1.466 s (10672528 allocations: 1.77 GiB)