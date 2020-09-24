
@views function get_phi_E(t, c, d, gp::GaussianProcess)
	i_low = sum(1:c-1)
	i_high = i_low + c
	
	G_pars_sub = gp.dpars.G[d, i_low+1:i_high, :]
	
	hcat(map.((pi -> map.(ppi -> gp.base_kernel(t, t, pi, ppi, gp.dpars.u), 
					G_pars_sub)), G_pars_sub)...)
end 

@views function get_phi_cov(t, tp, c, cp, d, dp, gp::GaussianProcess)
	i_low = sum(1:c-1)
	i_high = i_low + c
	
	ip_low = sum(1:cp-1)
	ip_high = ip_low + cp 
	
	G_pars_sub = [gp.dpars.G[d, i_low+1:i_high, :] ; gp.dpars.G[dp, ip_low+1:ip_high, :]]
	cat(map.((pi -> map.(ppi -> gp.base_kernel(t, tp, pi, ppi, gp.dpars.u), 
					G_pars_sub)), G_pars_sub)..., dims=2)
end 

"""
calculates the expectation of products of Gaussian RVs, with
exponents s and covariance matrix phi, using the identity in proposition 1
of Kan 2008.
"""
function kan_rv_prod(phi::Array{Float64, 2}, s::Array{Int64, 1})
	
	st = sum(s)
	prods = map(x -> 0:x, s)
	E = 0.
	for v in Iterators.product(prods...)
		vl = collect(v)
		h = map((vi, si) -> si / 2 - vi, vl, s)
		
		coeff = (-1.0)^(sum(vl)) * mapreduce(binomial, *, s, vl)
 		hKh =  ( 0.5 * h' * phi * h)^(st / 2)
		E +=  coeff * hKh
	end
	E /= factorial(st ÷ 2)
end

function full_E(t, d, gp::GaussianProcess)
	val = 0
	for c in 1:gp.C
		if c % 2 == 0
			s = ones(Int64, c)
            phi = get_phi_E(t, c, d, gp)
			val += kan_rv_prod(phi, s)
		end
	end
	val 
end 

function full_cov(t, tp, d, dp, gp::GaussianProcess)
	val = 0
    for c in 1:gp.C
		for cp in 1:gp.C
			if (c + cp) % 2 == 0
				s = ones(Int64, c + cp)
                K = get_phi_cov(t, tp, c, cp, d, dp, gp)
				val += kan_rv_prod(K, s)
			end 
		end 
	end 
	val 
end 

function kernel(t, tp, d, dp, gp::GaussianProcess)
	cov = full_cov(t, tp, d, dp, gp)
    E = full_E(t, d, gp)
	Ep = full_E(tp, dp, gp)
	cov - E*Ep
end 	

