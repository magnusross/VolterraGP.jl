"""
Normalised mean square error on predictions 
"""
function NMSE(test::Data, gp::GaussianProcess)
    yp, _ = posterior(test.X, gp)
    y = reduce(vcat, test.Y)
    
    var_y = sum((y .- (sum(y) / 200)).^2)
    N = size(y)[1]

    (100 / (var_y * N)) * sum((yp .- y).^2)
end 