using Revise
using VolterraGP
using Flux
D = 3 # number of outputs 
C = 2 # order of volterra series 
P = 1 # number of paramters in base kernel 

train, test = generate_toy_data(20) # genrates toy data from paper 
gp = GaussianProcess(threeEQs, D, C, P, train) 

t = collect(0.0:0.1:1.)
Gs = collect(0.0:0.1:1.)

g = gradient(x -> sum(VolterraGP.get_phi_cov(0.1, 0.3, 2, 2, 1, 1, x)), gp)
# g[1]

b = gradient(x -> sum(VolterraGP.fill_phi(threeEQs, t, x, [0.1])), Gs)

println(gradient(VolterraGP.full_E, 0.1, 2, gp))
println(gradient(VolterraGP.full_cov, 0.1, 0.2, 2, 2, gp))
gradient(VolterraGP.kernel, 0.1, 0.1, 3, 2, gp)
