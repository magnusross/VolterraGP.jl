using Revise
using VolterraGP


D = 3 # number of outputs 
C = 1 # order of volterra series 
P = 2 # number of paramters in base kernel 

reps = 5
res = fill(0., reps)
for i = 1:reps
    train, test = generate_toy_data(50) # genrates toy data from paper 
    gp = GaussianProcess(scaledEQs, D, C, P, train) 
    fit!(gp, 30, ls_lr=5e-2, σ_lr=1e-2, show_like=true)
    res[i] = NMSE(test, gp)
end
 
m = sum(res) / reps 
std =  sqrt(sum((res .- m).^2) / reps)
println("NMSE: $m ± $std ")