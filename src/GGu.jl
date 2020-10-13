"""
calculated this by hand so may not be right,
all kernels (both smoothing and base process) in EQ form, see eqn 2
"""
function threeEQs(t::Float64, tp::Float64, Gp1::Float64, Gp2::Float64, bp::Array{Float64,1})::Float64
	a, b, c = Gp1, Gp2, bp[1]
	coeff = 1 / (sqrt(2 * π) * sqrt(abs(a^2 + b^2 + c^2)))
	ans =  exp(-0.5 * (1 / (a^2 + b^2 + c^2)) * (t - tp)^2) * coeff
	# ans > eps() ? ans : 0
	ans
end 


