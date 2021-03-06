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
	using ForwardDiff
	using TensorCast
	using Tullio

	export Data
	export GaussianProcess, DiffableParameters
	export negloglikelihood, posterior, threeEQs, scaledEQs, plotgp, generate_toy_data, fit! 
	export NMSE

	include("gp.jl")
	include("GGu.jl")
	include("volterra.jl")
	include("plot.jl")
	include("data/load.jl")
	include("fit.jl")
	include("utils.jl")
	include("metrics.jl")
end



