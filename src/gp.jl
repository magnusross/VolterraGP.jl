function init_G_pars(D, C, P)
	rand(Float64, (D, sum(1:C+1), P))
end

function K_from_kernel(t, tp, kernel, pars)
	cat(map.((tpi -> map.(ti -> kernel(ti, tpi, pars...), t)), tp)..., dims=2)
end

# ╔═╡ d63d349e-f8e3-11ea-2779-4bc400527671
function μ_from_full_E(t, pars)
	map.(ti -> full_E(ti, pars...), t)
end

# ╔═╡ 0c6b16b6-f994-11ea-1edb-3b62ddf90604
function GP_posterior(t_obs, y_obs, t_pred, noise, kernel, pars)
	Koo = K_from_kernel(t_obs, t_obs, kernel, pars) + noise[1]^2*I
	Kop = K_from_kernel(t_obs, t_pred, kernel, pars)
	Kpp = K_from_kernel(t_pred, t_pred, kernel, pars) + 1e-8*I
	
	μo = μ_from_full_E(t_obs, pars)
	μp = μ_from_full_E(t_pred, pars) 
	
	Loo = cholesky(Koo).L 

	μ_post = μp + Kop'*(Loo'\(Loo\(y_obs - μo)))
	
	Lop = Loo\Kop
	K_post = Kpp - Lop'*Lop
	
	(μ_post, K_post)
end


# ╔═╡ ece25e0e-f997-11ea-38c4-5ba3e323afe5
function GP_log_likelihood(t_obs, y_obs, noise, kernel, pars)
	K = K_from_kernel(t_obs, t_obs, kernel, pars)
	μ = μ_from_full_E(t_obs, pars)
	Kp = K + noise[1].^2*I
	0.5 * ( y_obs'*inv(Kp)*y_obs + log(det(Kp)) + size(y_obs)[1]*log(2*π))
end
