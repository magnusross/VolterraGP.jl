module VolterraGP
	using Base
	using LinearAlgebra
	using Random
	using Distributions
	using Plots

	export init_G_pars, GP_posterior, GP_log_likelihood, kernel, GGu_cov
	include("gp.jl")
	include("GGu.jl")
	include("volterra.jl")
end



