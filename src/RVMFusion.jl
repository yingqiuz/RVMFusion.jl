#  RVM for lower quality data
module RVMFusion
using Statistics, LinearAlgebra, StatsBase, Distributions
using ThreadsX, Transducers, Folds, ProgressMeter
import LoopVectorization

export RVM, RVM!, sigmoid, softmax

include("utils.jl")
include("binomial.jl")
include("multinomial.jl")

function RVM(X::Matrix{T}, t::Vector{Int64}, α_init::Float64=1.0;
             kw...) where T<:Real
    n = size(X, 1)
    d = size(X, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))

    K = size(unique(t), 1)
    if K > 2
        α = ones(T, d, K) .* α_init
        t2 = [j == t[i] for i ∈ 1:n, j ∈ 1:K]
        RVM!(X, convert(Matrix{T}, t2), α; kw...)
    elseif K == 2
        α = ones(T, d) .* α_init
        RVM!(X, convert(Vector{T}, t), α; kw...)
    else
        throw(TypeError("Number of classes less than 2."))
    end
end

function RVM(
    XH::Matrix{T}, XL::Matrix{T}, t::Vector{Int64},
    XLtest::Matrix{T}, α_init::Float64=1.0,
    β_init::Float64=1.0; kw...
) where T<:Real
    n = size(XH, 1)
    d = size(XH, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))

    K = size(unique(t), 1)
    if K > 2
        α = ones(T, d, K) .* α_init
        β = ones(T, d, K) .* β_init
        t2 = [j == t[i] for i ∈ 1:n, j ∈ 1:K]
        RVM!(XH, XL, convert(Matrix{T}, t2), XLtest, α, β; kw...)
    elseif K == 2
        α = ones(T, d) .* α_init
        β = ones(T, d) .* β_init
        RVM!(XH, XL, convert(Vector{T}, t), XLtest, α, β; kw...)
    else
        throw(TypeError("Number of classes less than 2."))
    end
end


end
