"""
only works when t's all the same for outputs 
"""
function plotgp(t::Array{Float64}, gp::GaussianProcess; N=50, jitter=1e-5)

    rs = x -> reshape(x, (size(t)[1], gp.D))

    μ, K = posterior(t, gp)
    # dist = MvNormal(μ, K + I)
    μ_arr = rs(μ)
    K_arr = rs(sqrt.(diag(K + jitter * I)))


    plot(t, μ_arr, legend=false, layout=gp.D)
    plot!(t, μ_arr + 2 * K_arr, lc="red", layout=(1, gp.D))
    plot!(t, μ_arr - 2 * K_arr, lc="red", layout=(1, gp.D))
    scatter!(gp.data.X, hcat(gp.data.Y...),
        layout=(1, gp.D), 
        markershape=:o,
        markercolor="green",
        markersize=2,
        markerstrokewidth=0)
end 

