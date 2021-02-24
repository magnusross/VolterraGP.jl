using Flux
using Flux:@adjoint

function kan_rv_prod_inner(phi::Array{AbstractFloat,2}, v::NTuple)
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
function kan_rv_prod(phi)
	st = size(phi)[1]
	mapreduce(v -> kan_rv_prod_inner(phi, v), +, Iterators.product(fill(0:1, st)...)) / factorial(st ÷ 2)
end

function kan_rv_prod2(phi)
    kan_rv_prod(phi)
end

function kan_rv_prod2_adj(phi)
    st = size(phi)[1]
    g = v -> gradient(kan_rv_prod_inner, phi, v)[1]
    sum(g, Iterators.product(fill(0:1, st)...)) / factorial(st ÷ 2)
    # gradient(kan_rv_prod, phi)[1]
end

@adjoint function kan_rv_prod2(phi)
    kan_rv_prod2(phi), x -> (x * kan_rv_prod2_adj(phi),)
end

println(kan_rv_prod2_adj(3 * ones(5, 5)))
# gradient(kan_rv_prod_inner, ones(2, 2), (1, 0))
# print(gradient(kan_rv_prod, ones((2, 2)))[1] - gradient(kan_rv_prod2, ones((2, 2)))[1])
"""
@btime VolterraGP.kan_rv_prod(ones((5, 5)))

@btime gradient(kan_rv_prod, ones((2, 2)))

@btime gradient(kan_rv_prod2, ones((2, 2)))

"""
# function sum2(op, arr)
# 	return sum(op, arr)
# end

# function sum2adj(Δ, op, arr)
# 	n = length(arr)
# 	g = x -> Δ * gradient(op, x)[1]
# 	return (nothing, map(g, arr))
# end

# @adjoint function sum2(op, arr)
# 	return sum2(op, arr), Δ -> sum2adj(Δ, op, arr)
# end

# function t1(arr, op)
# 	g = sum2(op, arr)
# 	return g
# end

# function t0(arr, op)
# 	g = sum(op, arr)
# 	return g
# end

# function testPerf()
	# nb = 100000
# 	arr = rand(nb)
# 	op = x -> x + x

# 	res = t1(arr, op)
# 	println(res)


# 	df = x -> gradient(x -> t1(x, op), x)
# 	res1 = df(arr)[1]

# 	df0 = x -> gradient(x -> t0(x, op), x)
# 	res0 = df0(arr)[1]

# 	diff = sum((res0 - res1).^2)
# 	println("diff : ")
# 	println(diff)

# 	println("Benchmarking t1")
# 	@time for i = 1:100
# 		t1(arr, op)
# 	end

# 	println("Benchmarking gradient")
# 	@time for i = 1:100
# 		df(arr)
# 	end

# end

# testPerf()

