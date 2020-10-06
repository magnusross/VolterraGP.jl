using Revise
using VolterraGP
using LinearAlgebra
using Plots


D = 3
C = 1 
P = 1
data = Data(collect(0.:0.1:1.), fill(collect(0.:0.01:1.), D))
for i in [0.0001, 0.001, 0.01, 0.1, 1.]
    gp =  i * ones(Float64, (D, sum(1:C), P))
    # gp[2, 1, 1] *= i
    # gp[1, 3, 1] += i

    dpars = DiffableParameters(fill(0.001, D),  gp,  [0.001])


    gp = GaussianProcess(threeEQs, D, C, P, data, dpars) 

    Σ = Diagonal(vcat([gp.dpars.σ[i]^2 * ones(size(gp.data.X)[1]) for i in 1:gp.D]...))
    
    K = VolterraGP.fill_K(data.X, data.X, gp) + Σ + 1e-5 * I
    plt = heatmap(K)
    display(plt)
    println(minimum(K))
    # println(maximum(K))
    println(minimum(eigvals(K)))
    println(maximum(K' - K))
end 

