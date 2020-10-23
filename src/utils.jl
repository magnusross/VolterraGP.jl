"""
splits long vector with all the outputs concatented into array of 
vectors of individule outputs 
"""
function split_outputs(y::Array{Float64}, t::Array{Array{Float64,1},1})
    s = [size(ti)[1] for ti in t]
    out = fill(Float64[], size(t)[1])
    out[1] = y[1:s[1]]
    sc = s[1]
    for i = 2:size(t)[1]
        out[i] = y[sc + 1:sc + s[i]]
        sc += s[i]
    end
    out
end 

sample_mean(s) = s / size(s)[1]
sample_std(s) = sqrt(sum((s .- sample_mean(s)).^2) / size(s)[1])

# function int_to_bit(u)
#     res = map(parse, bitstring(u))
#     # res.chunks[1] = u % UInt64
# end

# function gen_prod(c)
#     map(x -> int_to_bit(x)[1:c], 1:Int64(2^c - 1))
# end



