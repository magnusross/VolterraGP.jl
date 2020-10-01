function threeEQs(t::Float64, tp::Float64, Gp1::Float64, Gp2::Float64, bp::Array{Float64,1})::Float64
	a, b, c = Gp1, Gp2, bp[1]
	coeff = 1. / (sqrt(2 * π * (a + b + c)^2))
	coeff * exp(-0.5 * (1 / (a + b + c)^2) * (t - tp)^2)
end 