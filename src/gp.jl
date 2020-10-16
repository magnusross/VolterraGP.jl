const rng = MersenneTwister(1234)

struct Data 
    X::Array{Float64,1} 
    Y::Array{Array{Float64,1},1} 
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

"""
initialise the differentiable paramters
"""
function init_dpars(D::Int64, C::Int64, P::Int64)::DiffableParameters
    G = 0.1 * ones(Float64, (D, sum(1:C), P))
    DiffableParameters(fill(0.3, D), G, [.01])
end

"""
get K for one output 
"""
function fill_sub_K(t::Array{Float64}, tp::Array{Float64}, d::Int64, dp::Int64, gp::GaussianProcess)::Array{Float64,2}
    K  = reduce(hcat,
        map.((tpi -> 
            map.(ti ->
                kernel(ti, tpi, d, dp, gp), 
            t)), 
        tp))
    # println(d, dp, " ", minimum(eigvals(K)))
    # Matrix(Hermitian(K))
    K
end

"""
get K for all outputs 
"""
function fill_K(t::Array{Float64}, tp::Array{Float64}, gp::GaussianProcess)
    reduce(vcat,
        map.(dpi -> 
            reduce(hcat, 
                map.(di -> 
                    fill_sub_K(t, tp, di, dpi, gp), 
                 1:gp.D)), 
        1:gp.D))
end 


"""
get mean for one output
"""
function fill_sub_μ(t::Array{Float64}, d::Int64, gp::GaussianProcess)::Array{Float64,1}
	map.(ti -> full_E(ti, d, gp), t)
end

"""
get mean for all outputs 
"""
function fill_μ(t::Array{Float64}, gp::GaussianProcess)::Array{Float64,1}
    reduce(vcat, map.(di -> fill_sub_μ(t, di, gp), 1:gp.D))
end


function posterior(t::Array{Float64}, gp::GaussianProcess; jitter=1e-5)::Tuple{Array{Float64,1},Array{Float64,2}}
   
    Σ = Diagonal(vcat([gp.dpars.σ[i]^2 * ones(size(gp.data.X)[1]) for i in 1:gp.D]...))

    Koo = fill_K(gp.data.X, gp.data.X, gp) + Σ + jitter * I

    if !ishermitian(Koo)
        ∇ = maximum(Koo' - Koo)
        if ∇ > sqrt(eps())
            print("WARNING, Hermitian check faliure not rounding error! ", ∇)
        end 
        Koo = Matrix(Hermitian(Koo))
    end
    
    Kop = fill_K(gp.data.X, t, gp)
    Kpp = fill_K(t, t, gp)
    
    μo = fill_μ(gp.data.X, gp)
    μp = fill_μ(t, gp)

    y = vcat(gp.data.Y...)
    
    Loo = cholesky(Koo).L
    
    μ_post = μp + Kop' * (Loo' \ (Loo \ (y - μo)))



	Lop = Loo \ Kop
	K_post = Kpp - Lop' * Lop
	K_post = Matrix(Hermitian(K_post))
	(μ_post, K_post) 
end



function negloglikelihood(gp::GaussianProcess; jitter=1e-5)::Float64
  
    Σ = Diagonal(vcat([gp.dpars.σ[i]^2 * ones(size(gp.data.X)[1]) for i in 1:gp.D]...))
    K = fill_K(gp.data.X, gp.data.X, gp) + Σ + jitter * I
    
    if !ishermitian(K)
        ∇ = maximum(K' - K)
        if ∇ > sqrt(eps())
            print("WARNING, Hermitian check faliure not rounding error! ", ∇)
        end 
        K = Matrix(Hermitian(K))
    end

    μ = fill_μ(gp.data.X, gp)
    

    y = vcat(gp.data.Y...)

    dist = MvNormal(μ, K)
    -1 * logpdf(dist, y)
end


