"""
makes phi matrix for calculating the mean function
i.e. the eqn below eqn 11
"""
@views function get_phi_E(t::Float64, c::Int64, d::Int64,
						 gp::GaussianProcess)::Array{Float64, 2}
	i_low = sum(1:c-1)
	i_high = i_low + c
	
	G_pars_sub = gp.dpars.G[d, i_low+1:i_high, :]
	
	reduce(hcat, 
				[[gp.base_kernel(t, t, pi, ppi, gp.dpars.u)
			for ppi in G_pars_sub]
		for	pi in G_pars_sub])
end 

"""
makes phi matrix for calculating the cross covariance
function i.e. the bit inside () just above example 2
"""
@views function get_phi_cov(t::Float64, tp::Float64, c::Int64, cp::Int64, 
							d::Int64, dp::Int64, gp::GaussianProcess)::Array{Float64, 2}
	i_low = sum(1:c-1) 
	i_high = i_low + c
	
	ip_low = sum(1:cp-1)
	ip_high = ip_low + cp 

	G_pars_sub = [gp.dpars.G[d, i_low+1:i_high, :] ; gp.dpars.G[dp, ip_low+1:ip_high, :]]
	
	reduce(hcat, 
				[[gp.base_kernel(t, tp, pi, ppi, gp.dpars.u)
			for ppi in G_pars_sub]
		for	pi in G_pars_sub])
 end 

"""
calcultes terms inside the sum for prop one kan 2008, as it was faster this way
"""
function kan_rv_prod_inner(phi::Array{Float64, 2}, v::NTuple)
	vl = collect(v)
	h = 0.5 .- vl
	hKh =  ( 0.5 * dot(h, phi*h))^(size(phi)[1] / 2)
	(-1.0)^(sum(vl)) * hKh 
end 

"""
calculates the expectation of products of Gaussian RVs, with
exponents s and covariance matrix phi, using the identity in proposition 1
of Kan 2008.
"""
function kan_rv_prod(phi::Array{Float64, 2})::Float64
	st = size(phi)[1]
	mapreduce(v -> kan_rv_prod_inner(phi, v), +, Iterators.product(fill(0:1, st)...)) / factorial(st ÷ 2)
end

"""
this is a custom adjoint for Zygote autodiff
https://github.com/FluxML/Zygote.jl/issues/292	
is a bit of a hack and needs properly testing, but seems to work 
and provided a 10x speedup 
"""
function kan_rv_prod_adj(phi::Array{Float64, 2})
    st = size(phi)[1]
    g = v -> gradient(kan_rv_prod_inner, phi, v)[1]
	sum(g, Iterators.product(fill(0:1, st)...)) / factorial(st ÷ 2)
end

"""
see kan_rv_prod_adj
"""
@adjoint function kan_rv_prod(phi::Array{Float64, 2})
    kan_rv_prod(phi), x -> (x * kan_rv_prod_adj(phi),)
end

"""
implements eqn 10
"""
function full_E(t::Float64, d::Int64, gp::GaussianProcess)::Float64
	val = 0.
	for c in 1:gp.C
		if c % 2 == 0
            phi = get_phi_E(t, c, d, gp)
			val += kan_rv_prod(phi)
		end
	end
	val 
end 

"""
implements eqn 12
"""
function full_cov(t::Float64, tp::Float64, d::Int64, dp::Int64, gp::GaussianProcess)::Float64
	val = 0.
    for c in 1:gp.C
		for cp in 1:gp.C
			if (c + cp) % 2 == 0
                K = get_phi_cov(t, tp, c, cp, d, dp, gp)
				val += kan_rv_prod(K)
			end 
		end 
	end 
	val 
end 


"""
implements equation just above section 3.2
"""
function kernel(t::Float64, tp::Float64, d::Int64, dp::Int64, gp::GaussianProcess)::Float64
	cov = full_cov(t, tp, d, dp, gp)
    E = full_E(t, d, gp)
	Ep = full_E(tp, dp, gp)
	cov - E*Ep
end 	

