"""
Implements toy data from alvarez2019non
"""
function generate_toy_data()

    X = collect(0.005:0.005:1.)
    Y = fill(Float64[], 3)

    Sl = [5, 1, 2]
    Pl = [200, 0.1, 100]

    conv_ker = (Δ, S, P) ->  S * exp(-P * Δ^2)
    u = t -> sum(k -> (1 / k^2) * cos(2 * k * π * t), 1:4)
    
    for i in 1:3
        y = fill(0., 200)
        for j in 1:200
            y[j] =  quadgk(τ -> conv_ker(X[j] - τ, Sl[i], Pl[i]) * u(τ), 0., X[j])[1]
        end
        y = sum(x -> y.^x, 1:3)

        Y[i] = y + 0.005 * randn(200) * sum(( y .- (sum(y) / 200).^2))
    end

    mix = randperm(200)
    mixte = sort(mix[1:50])
    mixtr = sort(mix[50:end])

    train_X = X[mixtr]
    test_X = X[mixte]
    train_Y = fill(Float64[], 3)
    test_Y = fill(Float64[], 3)
    for i in 1:3
 

        train_Y[i] = Y[i][mixtr]
        test_Y[i] = Y[i][mixte]
    end 


    (Data(train_X, train_Y), Data(test_X, test_Y))
end     