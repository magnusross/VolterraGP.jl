using Revise
using VolterraGP
using Plots
using Flux

D = 3
C = 1
P = 1

train, test = generate_toy_data()

gp = GaussianProcess(threeEQs, D, C, P, train)

opt = Flux.ADAM(0.05)
its = 30

# train
negloglikelihood(gp)

for i in 1:its
        grads = gradient(Flux.params(gp.dpars.σ, gp.dpars.G, gp.dpars.u)) do
                            negloglikelihood(gp)
        end
    for p in (gp.dpars.σ, gp.dpars.G, gp.dpars.u)
      Flux.Optimise.update!(opt, p, grads[p])
    end
    # print("negloglike: ", negloglikelihood(gp), "\n")

end 

plotgp(test.X, gp)