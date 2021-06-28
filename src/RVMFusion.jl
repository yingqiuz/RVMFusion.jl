#  RVM for lower quality data
module RVMFusion
using Statistics, LinearAlgebra, StatsBase, Distributions
using Transducers, Folds, ProgressMeter, LoopVectorization

export RVM, RVM!
export sigmoid, softmax, f1, predict
export RVModel, FusedRVModel
# define type
abstract type Model end

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
        RVM!(copy(X), convert(Matrix{T}, t2), α; kw...)
    elseif K == 2
        α = ones(T, d) .* α_init
        RVM!(copy(X), convert(Vector{T}, t), α; kw...)
    else
        throw(TypeError("Number of classes less than 2."))
    end
end

function RVM(
    XH::Matrix{T}, XL::Matrix{T}, t::Vector{Int64},
    α_init::Float64=1.0, β_init::Float64=1.0;
    kw...
) where T<:Real
    n = size(XH, 1)
    d = size(XH, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))

    K = size(unique(t), 1)
    if K > 2
        α = ones(T, d, K) .* α_init
        β = ones(T, d, K) .* β_init
        t2 = [j == t[i] for i ∈ 1:n, j ∈ 1:K]
        RVM!(copy(XH), copy(XL), convert(Matrix{T}, t2), α, β; kw...)
    elseif K == 2
        α = ones(T, d) .* α_init
        β = ones(T, d) .* β_init
        RVM!(copy(XH), copy(XL), convert(Vector{T}, t), α, β; kw...)
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
        RVM!(copy(XH), copy(XL), convert(Matrix{T}, t2), copy(XLtest), α, β; kw...)
    elseif K == 2
        α = ones(T, d) .* α_init
        β = ones(T, d) .* β_init
        RVM!(copy(XH), copy(XL), convert(Vector{T}, t), copy(XLtest), α, β; kw...)
    else
        throw(TypeError("Number of classes less than 2."))
    end
end

#binomial
function RVM!(
    XH::AbstractMatrix{T}, XL::AbstractMatrix{T},
    t::AbstractVector{T},
    α::AbstractVector{T}, β::AbstractVector{T};
    rtol::Float64=1e-6, atol::Float64=1e-6,
    maxiter::Int64=10000, n_samples::Int64=5000,
    BatchSize::Int64=size(XL, 1), BatchNorm=true#, StepSize::Float64=0.01
) where T<:Real
    model = RVM!(
        XH, t, α;
        rtol=rtol, atol=atol,
        maxiter=maxiter, BatchSize=BatchSize,
        BatchNorm=BatchNorm
    )
    RVM!(
        model, XL, t, α, β;
        rtol=rtol, atol=atol, maxiter=maxiter,
        n_samples=n_samples, BatchSize=BatchSize,
        BatchNorm=BatchNorm
    )
end

function RVM!(
    XH::AbstractMatrix{T}, XL::AbstractMatrix{T},
    t::AbstractVector{T}, XLtest::AbstractMatrix{T},
    α::AbstractVector{T}, β::AbstractVector{T};
    rtol::Float64=1e-6, atol::Float64=1e-6,
    maxiter::Int64=10000, n_samples::Int64=5000,
    BatchSize::Int64=size(XL, 1), BatchNorm=true#, StepSize::Float64=0.01
) where T<:Real
    model = RVM!(
        XH, t, α;
        rtol=rtol, atol=atol,
        maxiter=maxiter, BatchSize=BatchSize,
        BatchNorm=BatchNorm
    )
    RVM!(
        model, XL, t, XLtest, α, β;
        rtol=rtol, atol=atol, maxiter=maxiter,
        n_samples=n_samples, BatchSize=BatchSize,
        BatchNorm=BatchNorm
    )
end

# multinomial
"""train only"""
function RVM!(
    XH::AbstractMatrix{T}, XL::AbstractMatrix{T}, t::AbstractMatrix{T},
    α::AbstractMatrix{T}, β::AbstractMatrix{T}; kw...
) where T<:Real
    model = RVM!(
        XH, t, α;
        rtol=rtol, atol=atol,
        maxiter=maxiter, BatchSize=BatchSize,
        BatchNorm=BatchNorm
    )
    RVM!(model, XL, t, α, β; kw...)
end

"""train + predict"""
function RVM!(
    XH::AbstractMatrix{T}, XL::AbstractMatrix{T}, t::AbstractMatrix{T},
    XLtest::AbstractMatrix{T}, α::AbstractMatrix{T}, β::AbstractMatrix{T};
    kw...
) where T<:Real
    model = RVM!(
        XH, t, α;
        rtol=rtol, atol=atol,
        maxiter=maxiter, BatchSize=BatchSize,
        BatchNorm=BatchNorm
    )
    RVM!(model, XL, t, XLtest, α, β; kw...)
end

end
