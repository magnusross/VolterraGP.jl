using VolterraGP: kan_rv_prod 
using Test

@testset "VolterraGP.jl" begin
    # Write your tests here.
end

@testset "volterra.jl" begin
    @test kan_rv_prod(ones(4, 4), ones(Int64, 4)) == 3.
    @test kan_rv_prod(ones(2, 2), ones(Int64, 2) .* 3) == 15.
end
