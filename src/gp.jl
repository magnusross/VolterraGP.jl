const rng = MersenneTwister(1234)

struct Data 
    X::Array{Array{<:AbstractFloat,1},1} 
    Y::Array{Array{<:AbstractFloat,1},1} 
    # add check here 
end

struct DiffableParameters
    σ::AbstractArray # measurement noise per output
    G::AbstractArray# all smoothing kenrel paramters 
    u::AbstractArray # u kernel paramters
end

mutable struct GaussianProcess
    base_kernel::Function
    D::Int # number of outputs 
    C::Int # terms in Volterra series 
    P::Int # Number of paramters in each smoothing kernel

    data::Union{Data,Missing} # data to fit 
    
    dpars::Union{DiffableParameters,Missing} # all the differentiable hyperparamters
end


GaussianProcess(base_kernel, D, C, P ) = GaussianProcess(base_kernel, D, C, P, missing, init_dpars(D, C, P))
GaussianProcess(base_kernel, D, C, P, data ) = GaussianProcess(base_kernel, D, C, P, data, init_dpars(D, C, P))


"""
initialise the differentiable paramters
"""
function init_dpars(D::Int, C::Int, P::Int)::DiffableParameters
    G = 0.1 * ones(Float64, (D, sum(1:C), P))
    DiffableParameters(fill(0.3, D), G, [.01])
end

"""
get K for one output 
"""
function fill_sub_K(t::Array{<:AbstractFloat}, tp::Array{<:AbstractFloat}, d::Int, dp::Int, gp::GaussianProcess)
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
function fill_K(t, tp, gp::GaussianProcess)
    reduce(hcat,
        map.(dpi -> 
            reduce(vcat, 
                map.(di -> 
                    fill_sub_K(t[di], tp[dpi], di, dpi, gp), 
                 1:gp.D)), 
        1:gp.D))
end 


"""
get mean for one output
"""
function fill_sub_μ(t::Array{<:AbstractFloat}, d::Int, gp::GaussianProcess)
	map.(ti -> full_E(ti, d, gp), t)
end

"""
get mean for all outputs 
"""
function fill_μ(t, gp::GaussianProcess)
    reduce(vcat, map.(di -> fill_sub_μ(t[di], di, gp), 1:gp.D))
end


function posterior(t, gp::GaussianProcess; jitter=1e-5)
    Σ = Diagonal(vcat([gp.dpars.σ[i]^2 * ones(size(gp.data.X[i])[1]) for i in 1:gp.D]...))

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



function negloglikelihood(gp::GaussianProcess)
  
    Σ = Diagonal(vcat([gp.dpars.σ[i]^2 * ones(size(gp.data.X[i])[1]) for i in 1:gp.D]...))
    K = fill_K(gp.data.X, gp.data.X, gp) + Σ + 1e-5 * I
    
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


function negloglikelihood(σ::AbstractArray, G::AbstractArray, u::AbstractArray, gp::GaussianProcess)
    gpn = copy(gp)
    gpn.dpars = DiffableParameters(σ, G, u)
    negloglikelihood(gpn)
end

