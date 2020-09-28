using Revise
using VolterraGP
using Plots


D = 3
C = 2
P = 1

# initialise
X = collect(Float64, -5:0.1:5)
Y = sin.(X) + 0.1*randn(size(X)[1])
data = Data(X, Y)

dpars = DiffableParameters([0.1], rand(Float64, (D, sum(1:C), P)), [0.1])
gp = GaussianProcess(threeEQs, D, C, P, data, dpars) 

K = VolterraGP.fill_K(X, X, gp)

heatmap(1:size(K)[1], 1:size(K)[1], K)