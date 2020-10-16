"""
only works when t's all the same for outputs 
"""
function plotgp(t::Array{Array{Float64,1},1} , gp::GaussianProcess; 
                samps=false, test=nothing, jitter=1e-6, save=nothing)

    rs(x) = begin
        s = [size(ti)[1] for ti in t]
        out = fill(Float64[], gp.D)
        out[1] = x[1:s[1]]
        sc = s[1]
        for i = 2:gp.D
            out[i] = x[sc + 1:sc + s[i]]
            sc += s[i]
        end
        out
    end 

    μ, K = posterior(t, gp)

    dist = MvNormal(μ, K + jitter * I)

    μ_arr = rs(μ)
    
    K_arr = rs(sqrt.(diag(K + jitter * I)))

    p = plot(t, μ_arr, legend=true, layout=gp.D)
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
            plot!(t, s, legend=false, lc="gray", lw=0.2)
        end
        
    end

    if save !== nothing
        savefig("/Users/magnus/Documents/phd/code/repos/VolterraGP/bin/plots/$save.svg")
    end
    display(p)
end

