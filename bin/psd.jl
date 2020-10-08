using Revise
using VolterraGP
using LinearAlgebra
using Plots

function diag_blocks!(K, nx, d)
    s = nx * d
    for i in 1:s
        for j in 1:s
            if (i - 1) ÷ nx !== (j - 1) ÷ nx 
                K[i, j] = 0
            end
        end 
    end 
end

D = 1
C = 2
P = 1

data = Data(collect(0.:0.3:1.), fill(collect(0.:0.01:1.), D))

for i in [0.0001, 0.001, 0.01, 0.1, 1.]
    
    # run with C=2
    Gpars =  i * ones(Float64, (D, sum(1:C), P))
    
    # run with C=1
    # Gpars = ones(Float64, (D, sum(1:C), P))
    # Gpars[2, 1, 1] -= i


    dpars = DiffableParameters(fill(0.001, D),  Gpars,  [0.001])

    gp = GaussianProcess(threeEQs, D, C, P, data, dpars) 
    
    # noise per output
    Σ = Diagonal(vcat([gp.dpars.σ[i]^2 * ones(size(gp.data.X)[1]) for i in 1:gp.D]...))
    
    K = VolterraGP.fill_K(data.X, data.X, gp) + Σ  + 1e-5 * I
    println("Normal:")
    println(minimum(eigvals(K)), "\n")


    # diag_blocks!(K, 2, 2)
    # println("Block Diag:")
    # println(minimum(eigvals(K)))

    plt = heatmap(K)
    display(plt)
end 

