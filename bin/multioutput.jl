using Revise
using VolterraGP
using Plots
using Flux
using LinearAlgebra 


D = 2
C = 2
P = 1

# initialise
X = collect(Float64, -5:1.:5)
Y = fill(Float64[], D)

t = collect(Float64, -20:0.3:20)


for i = 1:D
    Y[i] = i * sin.(X) +  (i-1)*0.5 * randn(size(X)[1])
end 



data = Data(X, Y)

# dpars = DiffableParameters([0.1], ones(Float64, (D, sum(1:C), P)), [0.1])
gp = GaussianProcess(threeEQs, D, C, P, data) 

# K = VolterraGP.fill_K(t, t, gp)

# heatmap(K)




opt = Flux.ADAM(0.1)
its = 30

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

K = VolterraGP.fill_K(gp.data.X, gp.data.X, gp)
p = VolterraGP.posterior(t, gp)
# μ_arr = reshape(p[1], (size(t)[1], D))
# plot(t, μ_arr , layout=(2, 1))
# scatter!(X, hcat(data.Y...) , layout=(2,1))

plotgp(t, gp)