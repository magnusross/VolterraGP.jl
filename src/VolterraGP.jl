module VolterraGP
	using Base
	using LinearAlgebra
	using Random
	using Distributions
	using Plots
	using QuadGK
	using Flux
	using Flux:@adjoint

	export Data
	export GaussianProcess, DiffableParameters
	export negloglikelihood, posterior1D, threeEQs, plotgp, generate_toy_data  
	
	include("gp.jl")
	include("GGu.jl")
	include("volterra.jl")
	include("plot.jl")
	include("data.jl")
end



