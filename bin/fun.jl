using VolterraGP
using Flux
using Distributions
using Plots
using LinearAlgebra

l=200
D, C, P = 1, 4, 1
G_pars = init_G_pars(D, C, P)
base_pars = [0.001]
pars = (GGu_cov, D, D, C, G_pars, base_pars)
noise = [0.1]
t = range(-12, stop=12, length=l)
t_obs = -6:0.5:6
y_obs = sin.(t_obs)

opt_G_pars = init_G_pars(D, C, P)
opt_base_pars = [1.]
opt_noise = [1.]
opt = Flux.ADAM(0.1)


   
# Learning Rate
its = 10
for i in 1:its
        grads = gradient(Flux.params(opt_noise, opt_G_pars, opt_base_pars)) do
     GP_log_likelihood(t_obs, y_obs, opt_noise, kernel, (GGu_cov, D, D, C, opt_G_pars, opt_base_pars))
        end
    for p in (opt_noise, opt_G_pars, opt_base_pars)
      Flux.Optimise.update!(opt, p, grads[p])
    end
    print(GP_log_likelihood(t_obs, y_obs, opt_noise, kernel, (GGu_cov, D, D, C, opt_G_pars, opt_base_pars)), "\n")

end 


opt_pars = (GGu_cov, D, D, C, opt_G_pars, opt_base_pars)
μ, K = GP_posterior(t_obs, y_obs, t, opt_noise, kernel, opt_pars);
N=50
dist = MvNormal(μ, K)
x = ones((l, N))
for i in 1:N
    x[:, i] = rand(dist)
end 
plot(t, x, linealpha=0.2, lc="blue", legend=false)
plot!(t, μ)
plot!(t, μ+2*sqrt.(diag(K)), lc="red")
plot!(t, μ-2*sqrt.(diag(K)), lc="red")
scatter!(t_obs, y_obs, 
    markershape = :x,
    markercolor="yellow",
    markersize=5,
    markerstrokewidth = 100)