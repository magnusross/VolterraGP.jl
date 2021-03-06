function fwd_grad_like(gp)
    dσ = ForwardDiff.gradient(x -> negloglikelihood(x, gp.dpars.G, gp.dpars.u, gp), gp.dpars.σ)
    dG = ForwardDiff.gradient(x -> negloglikelihood(gp.dpars.σ, x, gp.dpars.u, gp), gp.dpars.G)
    du = ForwardDiff.gradient(x -> negloglikelihood(gp.dpars.σ, gp.dpars.G, x, gp), gp.dpars.u)
    (dσ, dG, du)
end 

function fit!(gp, its; fwd=true, ls_lr=2e-3, σ_lr=2e-3, show_like=false)
    
    opt_ls = Flux.NADAM(ls_lr)
    opt_σ = Flux.NADAM(σ_lr)

    for i in 1:its
    # Σ = Diagonal(vcat([gp.dpars.σ[i]^2 * ones(size(gp.data.X)[1]) for i in 1:gp.D]...))
    # B = VolterraGP.fill_K(gp.data.X, gp.data.X, gp) + Σ + 1e-5 * I
    # println(sort(eigvals(B)))
    # println(maximum(B' - /B))
        if fwd
            grads = fwd_grad_like(gp)

            Flux.Optimise.update!(opt_σ, gp.dpars.σ, grads[1])
            Flux.Optimise.update!(opt_ls, gp.dpars.G, grads[2])
            Flux.Optimise.update!(opt_ls, gp.dpars.u, grads[3])
            
        else #use reverse diff 
            grads = Flux.gradient(Flux.params(gp.dpars.σ, gp.dpars.G, gp.dpars.u)) do
                negloglikelihood(gp)
            end

           for p in (gp.dpars.G, gp.dpars.u)
               Flux.Optimise.update!(opt_ls, p, grads[p])
           end
   
            Flux.Optimise.update!(opt_σ, gp.dpars.σ, grads[gp.dpars.σ])
        end 

    
        println("it: ", i)

        # println("params: ", gp.dpars.G, gp.dpars.σ, gp.dpars.u)
        if show_like
            println(" negloglike:", negloglikelihood(gp))
        end 
        
        if its ÷ 5 == 0
            plotgp(gp.data.X, gp)
        end 
    end 
end 