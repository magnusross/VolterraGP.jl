const rng = MersenneTwister(1234)
struct Data 
    X::Array{Float64,1} 
    Y::Array{Array{Float64,1},1} # 
end

struct DiffableParameters
    σ::Array{Float64,1} # measurement noise per output
    G::Array{Float64,3} # all smoothing kenrel paramters 
    u::Array{Float64,1} # u kernel paramters
end

mutable struct GaussianProcess
    base_kernel::Function
    D::Int64 # number of outputs 
    C::Int64 # terms in Volterra series 
    P::Int64 # Number of paramters in each smoothing kernel

    data::Union{Data,Missing} # data to fit 

    μ::Union{Array{Float64,1},Missing} # not used currently 
    K::Union{Array{Float64,2},Missing} # not used currently 
    
    dpars::Union{DiffableParameters,Missing} # all the differentiable hyperparamters
end

GaussianProcess(base_kernel, D, C, P ) = GaussianProcess(base_kernel, D, C, P, missing, missing, missing, init_dpars(D, C, P))
GaussianProcess(base_kernel, D, C, P, data ) = GaussianProcess(base_kernel, D, C, P, data, missing, missing, init_dpars(D, C, P))
GaussianProcess(base_kernel, D, C, P, data, dpars) = GaussianProcess(base_kernel, D, C, P, data, missing, missing, dpars)


function init_dpars(D::Int64, C::Int64, P::Int64)::DiffableParameters
    G = ones(Float64, (D, sum(1:C), P))
    DiffableParameters(0.1 * ones(D), G, [0.1])
end


function fill_sub_K(t::Array{Float64}, tp::Array{Float64}, d::Int64, dp::Int64, gp::GaussianProcess)::Array{Float64,2}
    reduce(hcat,
        map.((tpi -> 
            map.(ti ->
                kernel(ti, tpi, d, dp, gp), 
            t)), 
        tp))
end

function fill_K(t::Array{Float64}, tp::Array{Float64}, gp::GaussianProcess)
    reduce(vcat,
        map.(dpi -> 
            reduce(hcat, 
                map.(di -> 
                    fill_sub_K(t, tp, di, dpi, gp), 
                 1:gp.D)), 
        1:gp.D))
end 


function fill_sub_μ(t::Array{Float64}, d::Int64, gp::GaussianProcess)::Array{Float64,1}
	map.(ti -> full_E(ti, d, gp), t)
end

function fill_μ(t::Array{Float64}, gp::GaussianProcess)::Array{Float64,1}
    reduce(vcat, map.(di -> fill_sub_μ(t, di, gp), 1:gp.D))
end


function posterior(t::Array{Float64}, gp::GaussianProcess; jitter=1e-2)::Tuple{Array{Float64,1},Array{Float64,2}}
   
    Σ = Diagonal(vcat([gp.dpars.σ[i]^2 * ones(size(gp.data.X)[1]) for i in 1:gp.D]...))

    Koo = fill_K(gp.data.X, gp.data.X, gp) + Σ   + jitter * I

    if !ishermitian(Koo)
        ∇ = maximum(Koo' - Koo)
        if ∇ > 5 * eps()
            print("WARNING, Hermitian check faliure not rounding error! ", ∇)
        end 
        Koo = Hermitian(Koo)
    end
    
    Kop = fill_K(gp.data.X, t, gp)
    Kpp = fill_K(t, t, gp)

    # print(minimum(eigvals(Koo)), minimum(eigvals(Kpp)), minimum(eigvals(Kpp)))
    # print(sort(eigvals(Koo)))
    # print("\n\n", maximum(Koo' - Koo))
    
    μo = fill_μ(gp.data.X, gp)
    μp = fill_μ(t, gp)

    y = vcat(gp.data.Y...)
    
    Loo = cholesky(Koo).L
    
    μ_post = μp + Kop' * (Loo' \ (Loo \ (y - μo)))



	Lop = Loo \ Kop
	K_post = Kpp - Lop' * Lop
	
	(μ_post, K_post) 
end



function negloglikelihood(gp::GaussianProcess; jitter=1e-6)::Float64
    Σ = Diagonal(vcat([gp.dpars.σ[i]^2 * ones(size(gp.data.X)[1]) for i in 1:gp.D]...))

    K = fill_K(gp.data.X, gp.data.X, gp) + Σ + jitter * I
    print(det(K))

    μ = fill_μ(gp.data.X, gp)
    

    y = vcat(gp.data.Y...)

	0.5 * ( (y - μ)' * inv(K) * (y - μ) + log(det(K)) + size(y)[1] * log(2 * π))
end

# function posterior1D(t::Array{Float64}, gp::GaussianProcess; jitter=1e-5)::Tuple{Array{Float64,1},Array{Float64,2}}
#     Koo = fill_sub_K(gp.data.X, gp.data.X, 1, 1, gp) + gp.dpars.σ[1]^2 * I
#     Kop = fill_sub_K(gp.data.X, t, 1, 1, gp)
#     Kpp = fill_sub_K(t, t, 1, 1, gp)

#     μo = fill_sub_μ(gp.data.X, 1, gp)
#     μp = fill_sub_μ(t, 1, gp)

#     Loo = cholesky(Koo).L 

# 	μ_post = μp + Kop' * (Loo' \ (Loo \ (gp.data.Y - μo)))
    
# 	Lop = Loo \ Kop
# 	K_post = Kpp - Lop' * Lop
    
# 	(μ_post, K_post) 
# end