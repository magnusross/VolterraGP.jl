"""
only works when t's all the same for outputs 
"""
function plotgp(t::Array{Float64}, gp::GaussianProcess; 
                samps=false, test=nothing, jitter=1e-6, save=nothing)

    rs = x -> reshape(x, (size(t)[1], gp.D))

    μ, K = posterior(t, gp)

    dist = MvNormal(μ, K + jitter * I)
    μ_arr = rs(μ)
    K_arr = rs(sqrt.(diag(K + jitter * I)))


    plot(t, μ_arr, legend=true, layout=gp.D)
    plot!(t, μ_arr + 2 * K_arr, legend=false, lc="red", ls=:dot)
    plot!(t, μ_arr - 2 * K_arr, legend=false, lc="red", ls=:dot)
    scatter!(gp.data.X, hcat(gp.data.Y...),
        markershape=:o,
        markercolor="blue",
        markersize=1,
        markerstrokewidth=0)

    if test !== nothing
        scatter!(test.X, hcat(test.Y...),
        markershape=:o,
        markercolor="green",
        markersize=1,
        markerstrokewidth=0)
    end

    if samps
        N = 20 
        for i = 1:N
            s = rand(dist)
            s = rs(s)
            p = plot!(t, s, legend=false, lc="gray", lw=0.2)
            display(p)
        end
        
    end

    if save !== nothing
        savefig("/Users/magnus/Documents/phd/code/repos/VolterraGP/bin/plots/$save.svg")
    end
    
end

