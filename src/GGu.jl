using ForwardDiff

"""
calculated this by hand so may not be right,
all kernels (both smoothing and base process) in EQ form, see eqn 2
"""
function threeEQs(t::Float64, tp::Float64, 
				  Gp1::AbstractArray, Gp2::AbstractArray, bp::AbstractArray)::Float64
	a, b, c = sq
	coeff = 1 / (sqrt(2 * π) * sqrt(abs(a^2 + b^2 + c^2)))
	ans =  exp(-0.5 * (1 / (a^2 + b^2 + c^2)) * (t - tp)^2) * coeff
	ans
end 

function scaledEQs(ts::AbstractVector, # like ϕ[i, j] := scaledEQs(ts[i], ts[j], Gs[i, :], Gs[j, :], u[:])
	Gs::AbstractMatrix, bp::AbstractVector)
    @tullio phi[i,j] := exp(-0.5 * (inv(Gs[i,1]^2 + Gs[j,1]^2 + bp[1]^2) * (ts[i] - ts[j])^2)) * (Gs[i,2] * Gs[j,2]) / (sqrt(2 * π * (Gs[i,1]^2 + Gs[j,1]^2 + bp[1]^2)))
end 

# function scaledEQs(t::Float64, tp::Float64, 
# 				   Gp1::AbstractArray, Gp2::AbstractArray, bp::AbstractArray)::Float64
# 	sq = Gp1[1]^2 + Gp2[1]^2 + bp[1]^2
# 	exp(-0.5 * (1 / (sq) * (t - tp)^2)) * (Gp1[2] * Gp2[2]) / (sqrt(2 * π * sq))
# end 

# function scaledEQs_adj(t::Float64, tp::Float64, 
# 					   Gp1::AbstractArray, Gp2::AbstractArray, bp::AbstractArray)
# 	sq = Gp1[1]^2 + Gp2[1]^2 + bp[1]^2
# 	f_val = scaledEQs(t, tp, Gp1, Gp2, bp)
# 	∇sq =  (t - tp)^2 / (2 * sq^2) * f_val - 0.5 * (1 / sq) * f_val
# 	(nothing, nothing, [2 * Gp1[1] * ∇sq, f_val / Gp1[2]], [2 * Gp2[1] * ∇sq, f_val / Gp2[2]], [2 * bp[1] * ∇sq])
# end 

# @adjoint function scaledEQs(t::Float64, tp::Float64, 
# 	Gp1::AbstractArray, Gp2::AbstractArray, bp::AbstractArray)
# 	scaledEQs(t, tp, Gp1, Gp2, bp), _ -> scaledEQs_adj(t, tp, Gp1, Gp2, bp)
# end 