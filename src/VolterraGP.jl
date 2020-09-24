module VolterraGP
	using Base
	using LinearAlgebra
	using Random
	using Distributions
	using Plots

	export Data
	export GaussianProcess, DiffableParameters
	export negloglikelihood, posterior1D, threeEQs  
	include("gp.jl")
	include("GGu.jl")
	include("volterra.jl")
end



