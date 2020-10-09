using Revise
using VolterraGP


D = 3 # number of outputs 
C = 2 # order of volterra series 
P = 1 # number of paramters in base kernel 

train, test = generate_toy_data(40) # genrates toy data from paper 
gp = GaussianProcess(threeEQs, D, C, P, train) 

negloglikelihood(gp)

# fit!(gp, 20, ls_lr=1e-2, σ_lr=1e-3, show_like=true) # fails when ls_lr > 5e-3
# plotgp(test.X, gp) 