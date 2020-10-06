using Revise
using VolterraGP


D = 3 # number of outputs 
C = 1 # order of volterra series 
P = 1 # number of paramters in base kernel 

train, test = generate_toy_data(30) # genrates toy data from paper 
gp = GaussianProcess(threeEQs, D, C, P, train) 

fit!(gp, 10, ls_lr=1e-2) # fails when ls_lr > 5e-3
plotgp(test.X, gp) 