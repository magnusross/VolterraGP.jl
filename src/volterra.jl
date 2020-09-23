
@views function get_phi_E(t, f, c, d, G_pars, base_pars)
	i_low = sum(1:c)
	i_high = i_low + c
	
	G_pars_sub = G_pars[d, i_low+1:i_high, :]
	
	hcat(map.((pi -> map.(ppi -> f(t, t, pi, ppi, base_pars), 
					G_pars_sub)), G_pars_sub)...)
end 

@views function get_phi_cov(t, tp, f, c, cp, d, dp, G_pars, base_pars)
	i_low = sum(1:c)
	i_high = i_low + c
	
	ip_low = sum(1:cp)
	ip_high = ip_low + cp 
	
	G_pars_sub = [G_pars[d, i_low+1:i_high, :] ; G_pars[dp, ip_low+1:ip_high, :]]
	cat(map.((pi -> map.(ppi -> f(t, tp, pi, ppi, base_pars), 
					G_pars_sub)), G_pars_sub)..., dims=2)
end 

"""
calculates the expectation of products of Gaussian RVs, with
exponents s and covariance matrix phi, using the identity in proposition 1
of Kan 2008.
"""
function kan_rv_prod(phi, s)
	
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

function full_E(t, f, d, dp, C, G_pars, base_pars)
	val = 0
	for c in 1:C
		if c % 2 == 0
			s = ones(Int64, c)
			phi = get_phi_E(t, f, c, d, G_pars, base_pars)
			val += kan_rv_prod(phi, s)
		end
	end
	val 
end 

function full_cov(t, tp, f, d, dp, C, G_pars, base_pars)
	val = 0
	for c in 1:C
		for cp in 1:C
			if (c + cp) % 2 == 0
				s = ones(Int64, c + cp)
				K = get_phi_cov(t, tp, f, c, cp, d, dp, G_pars, base_pars)
				val += kan_rv_prod(K, s)
			end 
		end 
	end 
	val 
end 

function kernel(t, tp, f, d, dp, C, G_pars, base_pars)
	cov = full_cov(t, tp, f, d, dp, C, G_pars, base_pars)
	E = full_E(t, f, d, dp, C, G_pars, base_pars)
	Ep = full_E(tp, f, dp, dp, C, G_pars, base_pars)
	cov - E*Ep
end 	

