function WoodburyInv!(
    g::AbstractArray{T},
    X::AbstractArray{T},
) where T <: Real
    nrow, ncol = size(X)
    if nrow < ncol
        rmul!(X, Diagonal(sqrt.(1 ./ g)))
        C = cholesky!(Hermitian(I + X * X'))
        rmul!(X, Diagonal(sqrt.(1 ./ g)))
        return Diagonal(1 ./ g) - X' * LinearAlgebra.inv!(C) * X
    else
        return LinearAlgebra.inv!(cholesky!(Hermitian(Diagonal(g) + X' * X)))
    end
end

function WoodburyInv!(
    d::AbstractArray{T},
    g::AbstractArray{T},
    X::AbstractArray{T}
) where T <: Real
    nrow, ncol = size(X)
    if nrow < ncol
        rmul!(X, Diagonal(sqrt.(1 ./ g)))
        C = cholesky!(Hermitian(I + X * X'))
        d .= (1 ./ g) - (1 ./ sqrt.(g)) .* diag(X' * LinearAlgebra.inv!(C) * X) .* (1 ./ sqrt.(g))
        #rmul!(X, Diagonal(sqrt.(1 ./ g)))
        #return Diagonal(1 ./ g) - X' * inv!(C) * X
    else
        d .= diag(LinearAlgebra.inv!(cholesky!(Hermitian(Diagonal(g) + X' * X))))  # \ I(ncol)
    end
    d
end

function add_diagonal!(X::AbstractArray{T}, d::AbstractVector{T}) where T<:Real
    m = size(X, 1)
    n = size(X, 2)
    m == n || throw(DimensionMismatch("X must be a sqaure matrix."))
    m = min(m, size(d, 1))
    LoopVectorization.@tturbo for k = 1:m
        @inbounds X[k, k] += d[k]
    end
    return X
end

sigmoid(x) = 1 / (1 + exp(-x))

softmax(x::AbstractArray) = (LoopVectorization.@avx exp.(x) ./ sum(exp.(x), dims=2))

function f1(y1, y2)
    K = unique(y2)
    f = zeros(Float64, size(K, 1))
    for k in K
        TP = count(y1 .== y2 .== k)
        precision = TP / (TP + count( (y1 .== k) .& (y2 .!= k)))
        recall = TP / (TP + count((y1 .!= k) .& (y2 .== k)))
        f[k] = 2* precision * recall /(precision + recall)
    end
    f
end
