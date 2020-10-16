"""
Implements toy data from paper 
"""
function generate_toy_data(N_train=50)

    X = fill(collect(0.005:0.005:1.), 3)
    Y = fill(Float64[], 3)

    Sl = [5, 1, 2]
    Pl = [200, 0.1, 100]

    conv_ker = (Δ, S, P) ->  S * exp(-P * Δ^2)
    u = t -> sum(k -> (1 / k^2) * cos(2 * k * π * t), 1:4)
    
    for i in 1:3
        y = fill(0., 200)
        for j in 1:200
            y[j] =  quadgk(τ -> conv_ker(X[i][j] - τ, Sl[i], Pl[i]) * u(τ), 0., X[i][j])[1]
        end
        y = sum(x -> y.^x, 1:3)
        Y[i] = y +  randn(200) * sqrt(0.005 * (sum((y .- (sum(y) / 200)).^2) / 200))
    end



    train_X = fill(Float64[], 3)
    test_X = fill(Float64[], 3)
    train_Y = fill(Float64[], 3)
    test_Y = fill(Float64[], 3)
    for i in 1:3
        mix = randperm(200)
        mixtr = sort(mix[1:N_train])
        mixte = sort(mix[N_train:end])
 
        train_X[i] = X[i][mixtr]
        test_X[i] = X[i][mixte]
        train_Y[i] = Y[i][mixtr]
        test_Y[i] = Y[i][mixte]
    end 


    (Data(train_X, train_Y), Data(test_X, test_Y))
end     

function load_weather(N_train)
    
end 