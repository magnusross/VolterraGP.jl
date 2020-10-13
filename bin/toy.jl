using Revise
using VolterraGP


D = 3 # number of outputs 
C = 2 # order of volterra series 
P = 1 # number of paramters in base kernel 

train, test = generate_toy_data(50) # genrates toy data from paper 
gp = GaussianProcess(threeEQs, D, C, P, train) 

println(negloglikelihood(gp))

# gradient(negloglikelihood, gp)

fit!(gp, 50, ls_lr=1e-3, σ_lr=1e-3, show_like=true) # fails when ls_lr > 5e-3
plotgp(test.X, gp, test=test, samps=true, save="50")