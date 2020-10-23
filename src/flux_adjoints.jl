"""
this is a custom adjoint for Zygote autodiff
https://github.com/FluxML/Zygote.jl/issues/292	
is a bit of a hack and needs properly testing, but seems to work 
and provided a 10x speedup 
"""
# function kan_rv_prod_adj(phi::Array{Float64,2})
#     st = size(phi)[1]
#     g = v -> gradient(kan_rv_prod_inner, phi, v)[1]
# 	sum(g, Iterators.product(fill(0:1, st)...)) / factorial(st ÷ 2)
# end

# @adjoint function kan_rv_prod(phi::Array{Float64,2})
#     kan_rv_prod(phi), x -> (x * kan_rv_prod_adj(phi),)
# end

# @adjoint function map_prod(f, c)
#     map_prod(f, c), δ -> (δ * map_prod(x -> gradient(f, x), c), nothing)
# end 
# function kan_rv_prod_E_adj(t, c, d, gp)
#     g = v -> gradient(x -> kan_rv_prod_inner_E(t, c, d, x, v) / factorial(c ÷ 2), gp)
#     map(g, Iterators.product(fill(0:1, c)...)) 
# end 

# @adjoint function kan_rv_prod_E(t, c, d, gp)
#     kan_rv_prod_E(t, c, d, gp), Δ -> (nothing, nothing, nothing, Δ .* kan_rv_prod_E_adj(t, c, d, gp))
# end

# @adjoint function scaledEQs(t::Float64, tp::Float64, 
#     Gp1::Array{Float64,1}, Gp2::Array{Float64,1}, bp::Array{Float64,1})
#     dGp1, dGp2, dpb = gradient((Gp1, Gp2, bp) -> scaledEQs(t, tp, Gp1, Gp2, bp), Gp1, Gp2, bp)
#     scaledEQs(t, tp, Gp1, Gp2, bp), Δ -> (nothing, nothing, Δ * dGp1, Δ * dGp2, Δ * dpb)
# end 