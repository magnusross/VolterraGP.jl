function GGu_cov(t, tp, Gp1, Gp2, bp)
	a, b, c = Gp1[1], Gp2[1], bp[1]
	coeff = 1 / (sqrt(2 * π * (a + b + c) ^ 2))
	coeff * exp(-0.5 * (1 / (a + b + c) ^ 2) * (t - tp) ^ 2)
end 