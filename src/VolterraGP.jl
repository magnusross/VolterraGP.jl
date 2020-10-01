module VolterraGP
	using Base
	using LinearAlgebra
	using Random
	using Distributions
	using Plots
	using Flux

	export Data
	export GaussianProcess, DiffableParameters
	export negloglikelihood, posterior1D, threeEQs, plotgp  
	include("gp.jl")
	include("GGu.jl")
	include("volterra.jl")
	include("plot.jl")
end



