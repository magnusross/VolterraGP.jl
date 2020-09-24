const MissingOrArrF64 = Union{Missing,Array{Float64}}
struct Data 
    X::Array{Float64}
    Y::Array{Float64}
end

struct DiffableParameters
    σ::Array{Float64,1}
    G::Array{Float64,3}
    u::Array{Float64,1}
end


mutable struct GaussianProcess
    base_kernel::Function
    D::Int64
    C::Int64 
    P::Int64

    data::Union{Data,Missing} 

    μ::Union{Array{Float64,1},Missing}
    K::Union{Array{Float64,2},Missing}
    
    
    dpars::Union{DiffableParameters,Missing}
end

GaussianProcess(base_kernel, D, C, P ) = GaussianProcess(base_kernel, D, C, P, missing, missing, missing, init_dpars(D, C, P))
GaussianProcess(base_kernel, D, C, P, data ) = GaussianProcess(base_kernel, D, C, P, data , missing, missing, init_dpars(D, C, P))
GaussianProcess(base_kernel, D, C, P, data, dpars) = GaussianProcess(base_kernel, D, C, P, data , missing, missing, dpars )




function init_dpars(D, C, P)
    G = rand(Float64, (D, sum(1:C), P))
    DiffableParameters([0.1], G, [0.1])
end


function fill_sub_K(t, tp, d, dp, gp::GaussianProcess)
	cat(map.((tpi -> map.(ti -> kernel(ti, tpi, d, dp, gp), t)), tp)..., dims=2)
end

function fill_K(t, tp, gp::GaussianProcess)
    gp.K = cat(map.((dpi -> map.(di -> fill_sub_K(t, tp, di, dpi, gp), 1:gp.D)), 1:gp.D)..., dims=2)
end 


function fill_sub_μ(t, d, gp::GaussianProcess)
	map.(ti -> full_E(ti, d, gp), t)
end

function fill_μ(t, gp::GaussianProcess)
    gp.μ = cat(map.(di -> fill_sub_μ(t, di, gp), 1:gp.D), dims=1)
end


function posterior1D(t, gp::GaussianProcess; jitter=1e-5)
    Koo = fill_sub_K(gp.data.X, gp.data.X, 1, 1, gp) + gp.dpars.σ[1]^2*I
    Kop = fill_sub_K(gp.data.X, t, 1, 1, gp)
    Kpp = fill_sub_K(t, t, 1, 1, gp)

    μo = fill_sub_μ(gp.data.X, 1, gp)
    μp = fill_sub_μ(t, 1, gp)

    Loo = cholesky(Koo).L 

	μ_post = μp + Kop' * (Loo' \ (Loo \ (gp.data.Y - μo)))
	
	Lop = Loo \ Kop
	K_post = Kpp - Lop' * Lop
	
	(μ_post, K_post) 
end

function negloglikelihood(gp::GaussianProcess)
    K = fill_sub_K(gp.data.X, gp.data.X, 1, 1, gp) + gp.dpars.σ[1]^2*I
    μ = fill_sub_μ(gp.data.X, 1, gp)

    Kp = K + gp.dpars.σ[1].^2 * I
	0.5 * ( gp.data.Y' * inv(Kp) * gp.data.Y + log(det(Kp)) + size(gp.data.Y)[1] * log(2 * π))
end
