# function fill_phi(f::Function, ts::Array{Float64,1}, Gs::AbstractArray, u::Array{Float64,1})
# 	s = size(ts)[1] 
# 	reduce(hcat, 
# 				[[f(ts[j], ts[i], Gs[j, :], Gs[i, :], u)
# 			for i in 1:s]
# 		for	j in 1:s])
# end


# """
# makes phi matrix for calculating the mean function
# i.e. the eqn below eqn 11
# """
# @views function get_phi_E(t::Float64, c::Int64, d::Int64,
# 						 gp::GaussianProcess)::Array{Float64,2}
# 	i_low = sum(1:c - 1)
# 	i_high = i_low + c
	
# 	G_pars_sub = gp.dpars.G[d, i_low + 1:i_high, :]
	
# # 	reduce(hcat, 
# # 				[[gp.base_kernel(t, t, pi, ppi, gp.dpars.u) 
# # 			for ppi in G_pars_sub]
# # 		for	pi in G_pars_sub])
# 	ts = fill(t, c)

# 	fill_phi(gp.base_kernel, ts, G_pars_sub, gp.dpars.u)
# end


# # function fill_phi_zip_adj(f, ts, Gs, u)
# # 	big_grad = 	reduce(hcat, 
# # 							[[collect(gradient(f, ti, tpi, pi, ppi, u))
# # 						for (tpi, ppi) in zip(ts, Gs)]
# # 					for	(ti, pi) in zip(ts, Gs)])
# # 	big_grad
# # end

# """
# makes phi matrix for calculating the cross covariance
# function i.e. the bit inside () just above example 2
# """
# @views function get_phi_cov(t::Float64, tp::Float64, c::Int64, cp::Int64, 
# 							d::Int64, dp::Int64, gp::GaussianProcess)::Array{Float64,2}
# 	i_low = sum(1:c - 1) 
# 	i_high = i_low + c
	
# 	ip_low = sum(1:cp - 1)
# 	ip_high = ip_low + cp 

# 	G_pars_sub = [gp.dpars.G[d, i_low + 1:i_high, :] ; gp.dpars.G[dp, ip_low + 1:ip_high, :]]
# 	ts = [fill(t, c) ; fill(tp, cp)]
# 	fill_phi(gp.base_kernel, ts, G_pars_sub, gp.dpars.u)
# end 



"""
calcultes terms inside the sum for prop one kan 2008, as it was faster this way
"""
function kan_rv_prod_inner_E(t::Float64, c::Int64, d::Int64, gp::GaussianProcess, vl)
	# print("a")
	h = 0.5 .- vl

	i_low = sum(1:c - 1)
	i_high = i_low + c

	Gs = gp.dpars.G[d, i_low + 1:i_high, :]

	qf = 0.
	for i = 1:c 
		for j = 1:c 
			qf += h[i] * gp.base_kernel(t, t, Gs[i, :], Gs[j, :], gp.dpars.u) * h[j]
        end 
	end
	(-1.0)^(sum(vl)) * ( 0.5 * qf)^(c / 2)
end



function kan_rv_prod_inner_cov(t, tp, c, cp, d, dp, gp, vl)
	h = 0.5 .- vl
	ct = c + cp

	i_low = sum(1:c - 1) 
	i_high = i_low + c
	
	ip_low = sum(1:cp - 1)
	ip_high = ip_low + cp
	
	Gs = gp.dpars.G[d, i_low + 1:i_high, :]
	Gsp = gp.dpars.G[dp, ip_low + 1:ip_high, :]
	# Gt = [Gs ; Gsp] # can change so dont need to allocate for this possibly
	# ts = [fill(t, c) ; fill(tp, cp)] # abd this

	qf = 0.
	for i = 1:ct
		for j = 1:ct 
			if i <= c && j <= cp
				qf += h[i] * gp.base_kernel(t, t, Gs[i, :], Gsp[j, :], gp.dpars.u) * h[j]
			elseif i > c && j > cp  
				qf += h[i] * gp.base_kernel(tp, tp, Gsp[i-c, :], Gsp[j-cp, :], gp.dpars.u) * h[j]
        	elseif i > c && j <= cp
				qf += h[i] * gp.base_kernel(tp, t, Gs[i-c, :], Gsp[j, :], gp.dpars.u) * h[j]
			else 
				qf += h[i] * gp.base_kernel(t, tp, Gs[i, :], Gsp[j-cp, :], gp.dpars.u) * h[j]
    		end
    	end 
	end 
	(-1.0)^(sum(vl)) * ( 0.5 * qf)^(ct / 2)
end 

    """
calculates the expectation of products of Gaussian RVs, with
exponents s and covariance matrix phi, using the identity in proposition 1
of Kan 2008.
"""


# function kan_rv_prod_E(t, c, d, gp)
# 	map(v -> kan_rv_prod_inner_E(t, c, d, gp, v), Iterators.product(fill(0:1, c)...)) / factorial(c ÷ 2)
# end

# function kan_rv_prod_cov(t, tp, c, cp, d, dp, gp)::Float64
# 	mapreduce(v -> kan_rv_prod_inner_cov(t, tp, c, cp, d, dp, gp, collect(v)), +, Iterators.product(fill(0:1, c + cp)...)) / factorial((c + cp) ÷ 2)
# end

# function kan_rv_prod_inner(phi::Array{Float64,2}, v::NTuple)
# 	vl = collect(v)
# 	h = 0.5 .- vl
# 	hKh =  ( 0.5 * dot(h, phi * h))^(size(phi)[1] / 2)
# 	(-1.0)^(sum(vl)) * hKh 
# end 

# function kan_rv_prod(phi::Array{Float64,2})::Float64
# 	st = size(phi)[1]
# 	mapreduce(v -> kan_rv_prod_inner(phi, v), +, Iterators.product(fill(0:1, st)...)) / factorial(st ÷ 2)
# end




    """
implements eqn 10
"""
    function full_E(t::Float64, d::Int64, gp::GaussianProcess)::Float64
	val = 0.
	for c in 1:gp.C
		if c % 2 == 0
			vst = Iterators.product([0:1 for i = 1:c]...)
			int_val = 0.
			for v in vst
				int_val += kan_rv_prod_inner_E(t, c, d, gp, v) 
			end
			val += int_val / factorial(c ÷ 2)
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
				vst = Iterators.product([0:1 for i = 1:(c + cp)]...)
				int_val = 0.
				for v in vst
					int_val += kan_rv_prod_inner_cov(t, tp, c, cp, d, dp, gp, v) 
				end
				val += int_val / factorial((c + cp) ÷ 2) 
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
	
	cov - E * Ep 
    end 	

