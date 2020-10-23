"""
calculated this by hand so may not be right,
all kernels (both smoothing and base process) in EQ form, see eqn 2
"""
function threeEQs(t::Float64, tp::Float64, 
					Gp1::AbstractArray, Gp2::AbstractArray, bp::Array{Float64,1})::Float64
	a, b, c = Gp1[1], Gp2[1], bp[1]
	coeff = 1 / (sqrt(2 * π) * sqrt(abs(a^2 + b^2 + c^2)))
	ans =  exp(-0.5 * (1 / (a^2 + b^2 + c^2)) * (t - tp)^2) * coeff
	ans
end 

function scaledEQs(t::Float64, tp::Float64, 
				   Gp1::AbstractArray, Gp2::AbstractArray, bp::Array{Float64,1})::Float64
	a, b, c = Gp1[1], Gp2[1], bp[1]
	coeff = (Gp1[2] * Gp2[2]) / (sqrt(2 * π) * sqrt(abs(a^2 + b^2 + c^2)))
	ans =  exp(-0.5 * (1 / (a^2 + b^2 + c^2)) * (t - tp)^2) * coeff
	ans
end 


