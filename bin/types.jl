using Revise
using VolterraGP
using Plots

D = 1
C = 4 
P = 1

X = collect(-6:0.5:6)
Y = sin.(X)
data = Data(X, Y)

gp = GaussianProcess(threeEQs, D, C, P, data)

μ, K = posterior1D(collect(-5:0.1:5), gp)

heatmap(1:size(K,1), 1:size(K,2), K)