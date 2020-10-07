### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ e7925da0-08b0-11eb-10d8-e37b1e342978
begin
	using Revise 
	using VolterraGP
	using Plots
	using Distributions
end

# ╔═╡ 8b6f1218-08b3-11eb-2db7-13b175c26316
import Pkg; Pkg.add(Pkg.PackageSpec(path="/Users/magnus/Documents/phd/code/repos/VolterraGP"))

# ╔═╡ 20e87940-08b1-11eb-3618-a92d2228cb3a
md"
Volterra Kernels
================

An iteractive notebook demonstrating the properties of Volterra kernels
"

# ╔═╡ ab3de578-08b4-11eb-1fe9-2f18a33760a0
function k(τ, τp, C, Gpars, upars; mout=false)
	pars = DiffableParameters([0.1], Gpars, upars)
	if mout
		gp = GaussianProcess(threeEQs, 2, C, 1, Data([], [[]]), pars)
		return VolterraGP.kernel(τp, τ, 1, 2, gp)
	else
		gp = GaussianProcess(threeEQs, 1, C, 1, Data([], [[]]), pars)
		return VolterraGP.kernel(τp, τ, 1, 1, gp)
	end
end

# ╔═╡ 71a180c4-08b7-11eb-09cd-adf1fdfa7f14
md"
C=2 case, so 3 smoothing kerenels and one base kerenel, all of EQ form.
"

# ╔═╡ adcce676-08b5-11eb-393e-f3e2ed46246a
@bind g1 html"<input type='range' step='0.01' min='0' max='3'>"

# ╔═╡ 997768e4-08b6-11eb-1131-9b5ab9222922
@bind g2 html"<input type='range' step='0.01' min='0' max='3'>"

# ╔═╡ 9977aba6-08b6-11eb-16bd-fd79436e020a
@bind g3 html"<input type='range' step='0.01' min='0' max='3'>"

# ╔═╡ 9977f3e0-08b6-11eb-08f1-f1c92afa2158
@bind u html"<input type='range' step='0.01' min='0' max='3'>"

# ╔═╡ 40b9b7fe-08b4-11eb-37bc-8ff804c302da
begin
	τ = collect(-5:0.1:5)
	C = 2 
	GparsC2 = reshape([g1, g2, g3], (1, 3, 1))
	uparsC2 = [u]
	yC2 = [k(t, 0., C, GparsC2, uparsC2) for t in τ]
# 	K = reduce(hcat, [[k(t, tp, C, GparsC2, uparsC2) for t in τ] for tp in τ])
# 	dist = MvNormal(K)
# 	samp = rand(dist)
# 	plot(samp)
	plot(τ, yC2)
end

# ╔═╡ 6be3f44e-08b7-11eb-3827-fbbf2fff84b8
md"
Adjustable C with with homogeneous smoothing kernels, for 1 output
"

# ╔═╡ d20b4a9e-08b7-11eb-3093-9572d4fdbeca
@bind Cv html"<input type='range' step='1' min='1' max='5'>"

# ╔═╡ d34dcca4-08b7-11eb-38b7-21de4471ae2b
@bind g html"<input type='range' step='0.01' min='0' max='3'>"

# ╔═╡ d4d813c4-08b7-11eb-2fd9-ad141b799b1a
@bind uCv html"<input type='range' step='0.01' min='0' max='3'>"

# ╔═╡ e65c08f8-08b7-11eb-1680-b535c915caf0
begin
	GparsCv = fill(g, (1, sum(1:Cv), 1))
	uparsCv = [uCv]
	yCv = [k(t, 0., Cv, GparsCv, uparsCv) for t in τ]
	plot(τ, yCv)
end

# ╔═╡ 04e644ca-08bb-11eb-1c44-79d43cef287f
md"
Adjustable C with with homogeneous smoothing kernels, for different ouputs
"

# ╔═╡ 8e408b6a-08b9-11eb-046a-cfe34a818fb5
@bind Cvd html"<input type='range' step='1' min='1' max='5'>"

# ╔═╡ 9d98bb00-08b9-11eb-26de-159d482901cd
@bind gd1 html"<input type='range' step='0.01' min='0' max='3'>"

# ╔═╡ a5dd7256-08b9-11eb-257a-7b599ac29688
@bind gd2 html"<input type='range' step='0.01' min='0' max='3'>"

# ╔═╡ add40150-08b9-11eb-3b17-17b9a10668e3
@bind uCvd html"<input type='range' step='0.01' min='0' max='3'>"

# ╔═╡ c348ddb2-08b9-11eb-0d70-57bb6e917392
begin
	GparsCvd = [fill(gd1, (1, sum(1:Cvd), 1)) ; fill(gd2, (1, sum(1:Cvd), 1))]
	uparsCvd= [uCvd]
	yCvd = [k(t, 0., Cvd, GparsCvd, uparsCvd; mout=true) for t in τ]
	plot(τ, yCvd)
end

# ╔═╡ Cell order:
# ╠═8b6f1218-08b3-11eb-2db7-13b175c26316
# ╠═e7925da0-08b0-11eb-10d8-e37b1e342978
# ╟─20e87940-08b1-11eb-3618-a92d2228cb3a
# ╠═ab3de578-08b4-11eb-1fe9-2f18a33760a0
# ╟─71a180c4-08b7-11eb-09cd-adf1fdfa7f14
# ╠═adcce676-08b5-11eb-393e-f3e2ed46246a
# ╟─997768e4-08b6-11eb-1131-9b5ab9222922
# ╟─9977aba6-08b6-11eb-16bd-fd79436e020a
# ╟─9977f3e0-08b6-11eb-08f1-f1c92afa2158
# ╠═40b9b7fe-08b4-11eb-37bc-8ff804c302da
# ╠═6be3f44e-08b7-11eb-3827-fbbf2fff84b8
# ╠═d20b4a9e-08b7-11eb-3093-9572d4fdbeca
# ╠═d34dcca4-08b7-11eb-38b7-21de4471ae2b
# ╠═d4d813c4-08b7-11eb-2fd9-ad141b799b1a
# ╠═e65c08f8-08b7-11eb-1680-b535c915caf0
# ╟─04e644ca-08bb-11eb-1c44-79d43cef287f
# ╠═8e408b6a-08b9-11eb-046a-cfe34a818fb5
# ╠═9d98bb00-08b9-11eb-26de-159d482901cd
# ╠═a5dd7256-08b9-11eb-257a-7b599ac29688
# ╠═add40150-08b9-11eb-3b17-17b9a10668e3
# ╠═c348ddb2-08b9-11eb-0d70-57bb6e917392
