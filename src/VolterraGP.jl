module VolterraGP
	using Base
	using LinearAlgebra
	using Random
	using Distributions
	using DistributionsAD
	using Plots
	using QuadGK
	using Flux
	using Flux:@adjoint

	export Data
	export GaussianProcess, DiffableParameters
	export negloglikelihood, posterior, threeEQs, plotgp, generate_toy_data, fit! 

	include("gp.jl")
	include("GGu.jl")
	include("volterra.jl")
	include("plot.jl")
	include("data.jl")
	include("fit.jl")
end



