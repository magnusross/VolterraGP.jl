using Revise
using VolterraGP


D = 3 # number of outputs 
C = 1 # order of volterra series 
P = 2 # number of paramters in base kernel 

reps = 5
nmse_l = fill(0., reps)
nlpd_l = fill(0., reps)
for i = 1:reps
    train, test = generate_toy_data(50) # genrates toy data from paper 
    gp = GaussianProcess(scaledEQs, D, C, P, train) 
    fit!(gp, 100, ls_lr=5e-2, σ_lr=1e-2, show_like=true)
    nmse_l[i] = NMSE(test, gp)
end
 
print("NMSE ")
VolterraGP.summary_stats(nmse_l)
plotgp(test.X, gp)
