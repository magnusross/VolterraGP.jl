using Revise
using VolterraGP
using Plots
using Flux
using LinearAlgebra 


D = 2
C = 2
P = 1

X = collect(Float64, -5:1.:5)
Y = fill(Float64[], D)

t = collect(Float64, -20:0.3:20)

for i = 1:D
    Y[i] = i * sin.(X) 
end 

data = Data(X, Y)

dpars = DiffableParameters([0.1, 0.1], ones(Float64, (D, sum(1:C), P)), [0.1])
gp = GaussianProcess(threeEQs, D, C, P, data, dpars) 

# @btime negloglikelihood(gp)

# @btime gradient(Flux.params(gp.dpars.σ, gp.dpars.G, gp.dpars.u)) do
#     negloglikelihood(gp)
# end