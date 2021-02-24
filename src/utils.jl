"""
splits long vector with all the outputs concatented into array of 
vectors of individule outputs 
"""
function split_outputs(y::Array{<:AbstractFloat}, t::Array{Array{<:AbstractFloat,1},1})
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

sample_mean(s) = sum(s) / size(s)[1]
sample_std(s) = sqrt(sum((s .- sample_mean(s)).^2) / size(s)[1])

function summary_stats(s)
    m = sample_mean(s)
    std = sample_std(s)
    println("$m ± $std ")
end 


Base.copy(gp::GaussianProcess) = GaussianProcess(gp.base_kernel, gp.D, gp.C, gp.P, gp.data, gp.dpars)