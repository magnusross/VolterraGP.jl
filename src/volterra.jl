# # [a + b for a in 1:5, b in 1:3]
# function fill_phi(f, ts, Gs, u)
# 	f_u = x -> f(x..., u)
# 	@cast ϕ[i, j] := f_u((ts[i], ts[j], Gs[i, :], Gs[j, :]))
# 	# mapreduce((tsi, Gsi) -> 
# 	# 	mapreduce((tsj, Gsj) -> 
# 	# 		f(tsi, tsj, Gsi, Gsj, u), 
# 	# 	vcat, ts, Gs), 
# 	# hcat, ts, Gs)
# 	# reduce(hcat, 
# 	# 			[[f(ts[j], ts[i], Gs[j, :], Gs[i, :], u)
# 	# 		for i in 1:s]
# 	# 	for	j in 1:s])
# 	# [f(ts[j], ts[], Gs[j, :], Gs[i, :], u) for i in 1:s, j in 1:s]
# end


"""
makes phi matrix for calculating the mean function
i.e. the eqn below eqn 11
"""
@views function get_phi_E(t::AbstractFloat, c::Int, d::Int,
						 gp::GaussianProcess)
	i_low = sum(1:c - 1)
	i_high = i_low + c
	
	Gs = gp.dpars.G[d, i_low + 1:i_high, :]
	
# 	reduce(hcat, 
# 				[[gp.base_kernel(t, t, pi, ppi, gp.dpars.u) 
# 			for ppi in G_pars_sub]
# 		for	pi in G_pars_sub])3
	ts = fill(t, c)
	gp.base_kernel(ts, Gs, gp.dpars.u)
end


# function fill_phi_zip_adj(f, ts, Gs, u)
# 	big_grad = 	reduce(hcat, 
# 							[[collect(gradient(f, ti, tpi, pi, ppi, u))
# 						for (tpi, ppi) in zip(ts, Gs)]
# 					for	(ti, pi) in zip(ts, Gs)])
# 	big_grad
# end

    """
makes phi matrix for calculating the cross covariance
function i.e. the bit inside () just above example 2
"""
@views function get_phi_cov(t::AbstractFloat, tp::AbstractFloat, c::Int, cp::Int, 
							d::Int, dp::Int, gp::GaussianProcess)
	i_low = sum(1:c - 1) 
	i_high = i_low + c
	
	ip_low = sum(1:cp - 1)
	ip_high = ip_low + cp 

	Gs = [gp.dpars.G[d, i_low + 1:i_high, :] ; gp.dpars.G[dp, ip_low + 1:ip_high, :]]
	ts = [fill(t, c) ; fill(tp, cp)]
	gp.base_kernel(ts, Gs, gp.dpars.u)
end 

    """
calcultes terms inside the sum for prop one kan 2008, as it was faster this way
"""
function kan_rv_prod_inner(phi::AbstractArray, v::NTuple)
	vl = collect(v)
	h = 0.5 .- vl
	hKh =  ( 0.5 * dot(h, phi * h))^(size(phi)[1] / 2)
	(-1.0)^(sum(vl)) * hKh 
end 

"""
calculates the expectation of products of Gaussian RVs, with
exponents s and covariance matrix phi, using the identity in proposition 1
of Kan 2008.
"""
function kan_rv_prod(phi::AbstractArray)
	st = size(phi)[1]
	mapreduce(v -> kan_rv_prod_inner(phi, v), +, Iterators.product(fill(0:1, st)...)) / factorial(st ÷ 2)
end

"""
this is a custom adjoint for Zygote autodiff
https://github.com/FluxML/Zygote.jl/issues/292	
is a bit of a hack and needs properly testing, but seems to work 
and provided a 10x speedup 
"""
function kan_rv_prod_adj(phi::Array{<:AbstractFloat,2})
    st = size(phi)[1]
    g = v -> gradient(kan_rv_prod_inner, phi, v)[1]
	sum(g, Iterators.product(fill(0:1, st)...)) / factorial(st ÷ 2)
end

@adjoint function kan_rv_prod(phi::Array{<:AbstractFloat,2})
    kan_rv_prod(phi), x -> (x * kan_rv_prod_adj(phi),)
end

const hs_fs = Dict{Int, Tuple{Matrix{Float64}, Vector{Float64}}}()
function get_hs_fs(st)
	get!(hs_fs, st) do 
		print(st)
        hs = reduce(hcat, [0.5 .- digits(n, base=2, pad=st) for n in 0:2^st - 1])
        fs = map(v -> 2iseven(sum(v))-1, [digits(n, base=2, pad=st) for n in 0:2^st-1]) ./ factorial(st ÷ 2)
        (hs, fs)
    end
end

function kan_rv_prod_t(phi)
	st = size(phi, 1)
	hs, fs = get_hs_fs(st)
	@tullio ds[k] := hs[i,k] * phi[i,j] * hs[j,k] nograd=hs
	@tullio out := fs[k] * ds[k]^(st/2) nograd=fs
end 



"""
implements eqn 10
"""
function full_E(t::Real, d::Int, gp::GaussianProcess)
	val = 0.
	for c in 1:gp.C
		if c % 2 == 0
			phi = get_phi_E(t, c, d, gp)
			# println(kan_rv_prod(phi))
			val += kan_rv_prod(phi)
		end
	end
	val 
end

"""
implements eqn 12
"""
function full_cov(t::Real, tp::Real, d::Int, dp::Int, gp::GaussianProcess)
	val = 0.
        for c in 1:gp.C
		for cp in 1:gp.C
			if (c + cp) % 2 == 0
				# println(c,cp)
				phi = get_phi_cov(t, tp, c, cp, d, dp, gp)
				# println(size(phi))
				val += kan_rv_prod(phi)
			end 
		end 
	end 
	val 
end 


"""
implements equation just above section 3.2
"""
function kernel(t::Real, tp::Real, d::Int, dp::Int, gp::GaussianProcess)
	cov = full_cov(t, tp, d, dp, gp)
    E = full_E(t, d, gp)
	Ep = full_E(tp, dp, gp)
	
	cov - E*Ep 
end 	

