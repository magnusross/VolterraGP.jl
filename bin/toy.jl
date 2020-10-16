using Revise
using VolterraGP


D = 3 # number of outputs 
C = 1 # order of volterra series 
P = 1 # number of paramters in base kernel 

train, test = generate_toy_data(50) # genrates toy data from paper 
gp = GaussianProcess(threeEQs, D, C, P, train) 

println(negloglikelihood(gp))

# gradient(negloglikelihood, gp)

fit!(gp, 30, ls_lr=1e-1, σ_lr=1e-2, show_like=true) # fails when ls_lr > 5e-3
plotgp(test.X, gp, samps=true)