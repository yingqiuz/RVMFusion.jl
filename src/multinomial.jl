# multiclass
function RVM!(
    X::AbstractArray{T}, t::AbstractMatrix{T}, α::AbstractArray{T},
    method::Symbol=:block; tol=1e-5, maxiter=100000
) where T<:Real
    # Multinomial
    n = size(X, 1)
    d = size(X, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))
    K = size(unique(t), 1)  # total number of classes
    size(α, 2) == K || throw(DimensionMismatch("Number of classes and size of α mismatch."))

    # initialise
    # preallocate type-II likelihood (evidence) vector
    llh2 = Vector{T}(undef, maxiter)
    fill!(llh2, -Inf)
    w = ones(T, d, K) * 0.00001
    #αp = ones(T, d, K)
    A, Y, logY = (Matrix{T}(undef, n, K) for _ = 1:3)
    #a, y = (Vector{T}(undef, n) for _ = 1:2)
    #Xt = transpose(X)
    #t2 = similar(t)
    for iter ∈ 2:maxiter
        ind = unique!([item[1] for item in findall( α .< 10000)])
        αtmp = copy(α[ind, :])
        wtmp = copy(w[ind, :])
        Xtmp = @view X[:, ind]
        #copyto!(αp, α)
        llh2[iter] = Logit!(
            wtmp, αtmp, Xtmp,
            t, tol, maxiter, A, Y, logY
        )
        for k = 1:K
            # update alpha - what is y?
            #@views mul!(a, X, wtmp[:, k])
            #y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
            @views WoodburyInv!(
                αtmp[:, k], αtmp[:, k],
                Diagonal(sqrt.(logY[:, k] .* (1 .- logY[:, k]))) * X[:, ind]
            )
            @views α[ind, k] .= (1 .- α[ind, k] .* αtmp[:, k]) ./ wtmp[:, k].^2
        end
        w[ind, :] .= wtmp
        # check convergence
        incr = abs((llh2[iter] - llh2[iter-1]) / llh2[iter-1])
        @info "iteration $iter" incr
        if incr < tol
            H = Array{T}(undef, ind, ind, K)
            for k = 1:K
                # update alpha - what is y?
                #@views mul!(a, X, wtmp[:, k])
                #y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
                @views H[:, :, k] .= WoodburyInv(
                    αtmp[:, k],
                    Diagonal(sqrt.(logY[:, k] .* (1 .- logY[:, k]))) * X[:, ind]
                )
                #@views α[ind, k] .= (1 .- α[ind, k] .* αtmp[:, k]) ./ wtmp[:, k].^2
                return w, H, ind
            end
        end
        #copyto!(αp, α)
    end
    @warn "Not converged after $(maxiter) steps. Results may be inaccurate."
    #return RVModel(w, α)
end

function Logit!(
    w::AbstractMatrix{T}, α::AbstractMatrix{T}, X::AbstractMatrix{T},
    T::AbstractVector{Int64}, tol::Float64, maxiter::Int64,
    A::AbstractMatrix{T}, Y::AbstractMatrix{T}, logY::AbstractMatrix{T}
) where T<:Real
    n = size(t, 1)
    d = size(X, 2)
    K = size(unique(t), 1) # number of classes
    #dk = d * Ks
    g, wp = (similar(w) for _ = 1:2)
    #A = Matrix{T}(undef, n, K); Y = similar(A); logY = similar(A)
    #wp = similar(w)
    #Δw = similar(g)
    llhp = -Inf
    mul!(A, X, w)
    logY .= A .- log.(sum(exp.(A), dims=2))
    Y .= exp.(logY)
    r = [0.00001]  # initial step size
    for iter = 2:maxiter
        # update gradient
        mul!(g, Xt, T .- Y)
        g .-= w .* α
        copyto!(wp, w)
        # update weights
        w .+= g .* r
        mul!(A, X, w)
        logY .= A .- log.(sum(exp.(A), dims=2))
        Y .= exp.(logY)
        # update likelihood
        llh = -0.5sum(α .* w .* w) + sum(T .* logY)
        while (llh - llhp < 0)  # line search
            g ./= 2
            w .= wp .+ g .* r
            mul!(A, X, w)
            logY .= A .- log.(sum(exp.(A), dims=2))
            Y .= exp.(logY)
            llh = -0.5sum(α .* w .* w) + sum(T .* logY)
        end
        if llh - llhp < tol
            #H = WoodburyInv!(α, (-Y .* (T .- Y)) * X)
            #fill!(H, 0)
            #for k = 1:K, i = 1:d
                #idx = d * (k-1) + i
                #for l = 1:K, j = 1:d
                    #idy = d * (l-1) + j
                    #k == l ? δ = 1 : δ = 0
                    #for nn = 1:n
                        #@inbounds H[idy, idx] += X[nn, i] * X[nn, j] *
                                              #Y[nn, k] * (δ - Y[nn, l])
                    #end
                #end
            #end
            #add_diagonal!(H, α[:])
            return llh
        end
        llhp = llh
        # update step size
        r .= sum((w .- wp) .* (g .- gp))
        r .= abs.(r) ./ sum((g .- gp) .^ 2)
        copyto!(gp, g)
    end
    @warn "not converged."
end
