using Revise
using VolterraGP
using Flux
using Flux:@adjoint


pars = (threeEQs, collect(1.:1.:5.), 0.1 * collect(1.:1.:5.), [0.01])

function fill_test(f, ts, Gs, u)
	reduce(hcat, 
				[[f(ts[j], ts[i], Gs[j], Gs[i], u)
			for i in 1:size(ts)[1]]
		for	j in 1:size(ts)[1]])
end

function fill_phi_zip(f, ts, Gs, u)
	reduce(hcat, 
				[[f(ti, tpi, pi, ppi, u)
			for (tpi, ppi) in zip(ts, Gs)]
		for	(ti, pi) in zip(ts, Gs)])
end

@adjoint function fill_phi_zip(f, ts, Gs, u)

    Gadj = zip_adj_G(f, ts, Gs, u)
    uadj = zip_adj_u(f, ts, Gs, u)

    fill_phi_zip(f, ts, Gs, u), Δ -> (nothing, nothing,   Δ .* Gadj, Δ .* uadj) 
end 


function zip_adj_G(f, ts, Gs, u)
    N = size(ts)[1]
    Gadj = zeros((N, N, N))
    for n = 1:N
        for i = 1:N
            # only do upper triangle cos of symmetry 
            for j = i:N
                if i == n && j == n
                    g = gradient(x -> f(ts[i], ts[j], x, x, u), Gs[i])[1]
                    Gadj[i, j, n], Gadj[j, i, n] = g, g  
                elseif i == n
                    g = gradient(x -> f(ts[i], ts[j], x, Gs[j], u), Gs[i])[1]
                    Gadj[i, j, n], Gadj[j, i, n] = g, g
                elseif j == n 
                    g = gradient(x -> f(ts[i], ts[j], Gs[i], x, u), Gs[j])[1]
                    Gadj[i, j, n], Gadj[j, i, n] = g, g
                end 
            end 
        end 
    end
    Gadj     
end

function zip_adj_u(f, ts, Gs, u)
    N = size(ts)[1]
    uadj = zeros((N, N, size(u)[1]))
    for i = 1:N
        for j = 1:N
            uadj[i, j, :] = gradient(x -> f(ts[i], ts[j], Gs[i], Gs[j], x), u)[1]
        end
    end
    uadj
end


# @btime Gadj = zip_adj_G(pars...)
# @btime uadj = zip_adj_u(pars...)
ftest(t, tp, Ga, Gb, u) = (Ga + Gb) * (t + tp) + Gb * Ga * (t * tp)^2 + u[1]

# pars = (, [1., 2.], [2., 3.], [5.])
# a = fill_phi_zip(pars...)
@btime gradient(x -> sum(fill_phi_zip(ftest, collect(0.0:0.1:1), x, [0.1])), collect(0.0:0.1:1))

@btime gradient(x -> sum(fill_test(ftest, collect(0.0:0.1:1), x, [0.1])), collect(0.0:0.1:1))

y, back = Flux.pullback(fill_test, ftest, [1., 2.], [1., 1.], [0.1])
