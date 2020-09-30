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

function kernel(t::Float64, tp::Float64, d::Int64, dp::Int64, gp::GaussianProcess)::Float64
	cov = full_cov(t, tp, d, dp, gp)
    E = full_E(t, d, gp)
	Ep = full_E(tp, dp, gp)
	cov - E*Ep
end 	

