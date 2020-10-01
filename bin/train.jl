using Revise
using VolterraGP
using Flux

D = 2
C = 2
P = 1

# initialise
X = collect(-5:1:5)
Y = sin.(X) + 0.1*randn(size(X)[1])
data = Data(X, Y)

dpars = DiffableParameters([0.1], ones(Float64, (D, sum(1:C), P)), [0.1])
gp = GaussianProcess(threeEQs, D, C, P, data, dpars) 


opt = Flux.ADAM(0.01)
its = 10

# train
for i in 1:its
        grads = gradient(Flux.params(gp.dpars.σ, gp.dpars.G, gp.dpars.u)) do
                            negloglikelihood(gp)
        end
    for p in (gp.dpars.σ, gp.dpars.G, gp.dpars.u)
      Flux.Optimise.update!(opt, p, grads[p])
    end
    print("negloglike: ", negloglikelihood(gp), "\n")

end 

# plot
plotgp(collect(-10:0.1:10), gp)