function plotgp(t::Array{Float64}, gp::GaussianProcess; N=50, jitter=1e-8)
    μ, K = posterior1D(t, gp)
    dist = MvNormal(μ, K+jitter*I)

    x = ones((size(t)[1], N))
    for i in 1:N
        x[:, i] = rand(dist)
    end 

    plot(t, x, linealpha=0.2, lc="blue", legend=false)
    plot!(t, μ)
    plot!(t, μ+2*sqrt.(diag(K)), lc="red")
    plot!(t, μ-2*sqrt.(diag(K)), lc="red")
    scatter!(gp.data.X, gp.data.Y, 
        markershape = :x,
        markercolor="yellow",
        markersize=5,
        markerstrokewidth = 100)
end 