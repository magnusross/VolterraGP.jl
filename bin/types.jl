using Revise
using VolterraGP
using Plots
using Flux

D = 1
C = 1 
P = 1

X = collect(-1:0.1:1)
Y = sin.(X)
data = Data(X, Y)

dpars = DiffableParameters([0.1], ones(Float64, (D, sum(1:C), P)), [0.1])

gp = GaussianProcess(threeEQs, D, C, P, data, dpars)

μ, K = posterior1D(collect(-5:0.1:5), gp)

# heatmap(1:size(K,1), 1:size(K,2), K)
negloglikelihood(gp)
opt = Flux.ADAM(0.01)
its = 10
print(gp.dpars.G)

for i in 1:its
        grads = gradient(Flux.params(gp.dpars.σ, gp.dpars.G, gp.dpars.u)) do
                            negloglikelihood(gp)
        end
    for p in (gp.dpars.σ, gp.dpars.G, gp.dpars.u)
      Flux.Optimise.update!(opt, p, grads[p])
    end
    print(negloglikelihood(gp), "\n")

end 
print(gp.dpars.G)