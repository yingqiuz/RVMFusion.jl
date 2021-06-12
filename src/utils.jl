function WoodburyInv!(
    g::AbstractArray{T},
    X::AbstractArray{T},
) where T <: Real

    nrow, ncol = size(X)
    if nrow < ncol
        covmat = X * transpose(X)

        rmul!(X, Diagonal(sqrt.(1 ./ g)))
        C = cholesky!(Hermitian(I + X * X'))
        rmul!(X, Diagonal(sqrt.(1 ./ g)))
        return Diagonal(1 ./ g) - X' * (C \ X)
    else
        return Hermitian(Diagonal(g) + X' * X) \ I(ncol)
    end
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
