using Revise
using VolterraGP
using Plots
using Flux

D = 1
C = 2 
P = 1

X = collect(-6:0.5:6)
Y = sin.(X)
data = Data(X, Y)

gp = GaussianProcess(threeEQs, D, C, P, data)

μ, K = posterior1D(collect(-5:0.1:5), gp)

# heatmap(1:size(K,1), 1:size(K,2), K)

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