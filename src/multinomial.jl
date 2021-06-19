# multiclass
function predict(
    X::AbstractArray{T}, w::AbstractMatrix{T},
    H::AbstractArray{T}, ind::AbstractArray{Int64}
) where T <: Real
    Xview = @view X[:, ind]
    K = size(w, 2)
    n = size(X, 1)
    A = Matrix{T}(undef, n, K)
    for k ∈ 1:K
        p = view(A, :, k)
        p .= diag(Xview * view(H, :, :, k) * transpose(Xview))
        p .= (1 .+ π .* p ./ 8).^(-0.5) .* (Xview * view(w, :, k))
    end
    return exp.(A) ./ sum(exp.(A), dims=2)
    #p .= 1 ./ (1 .+ exp.(-1 .* p))
end

function predict!(
    A::AbstractMatrix{T}, X::AbstractArray{T}, w::AbstractMatrix{T},
    H::AbstractArray{T}, ind::AbstractArray{Int64}
) where T <: Real
    Xview = @view X[:, ind]
    K = size(w, 2)
    n = size(X, 1)
    for k ∈ 1:K
        p = view(A, :, k)
        p = diag(Xview * H * transpose(Xview))
        p .= (1 .+ π .* p ./ 8).^(-0.5) .* (Xview * view(w, :, k))
    end
    A .= exp.(A) ./ sum(exp.(A), dims=2)
end

function RVM!(
    X::Array{T}, t::Matrix{T}, α::Matrix{T},
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
            WoodburyInv!(
                view(αtmp, :, k), α[ind, k],
                Diagonal(sqrt.(logY[:, k] .* (1 .- logY[:, k]))) * Xtmp
            )
            @views α[ind, k] .= (1 .- α[ind, k] .* αtmp[:, k]) ./ wtmp[:, k].^2
        end
        w[ind, :] .= wtmp
        # check convergence
        incr = abs((llh2[iter] - llh2[iter-1]) / llh2[iter-1])
        @info "iteration $iter" incr
        if incr < tol
            H = Array{T}(undef, d, d, K)
            for k = 1:K
                @views H[:, :, k] .= WoodburyInv!(
                    αtmp[:, k],
                    Diagonal(sqrt.(logY[:, k] .* (1 .- logY[:, k]))) * X[:, ind]
                )
                #@views α[ind, k] .= (1 .- α[ind, k] .* αtmp[:, k]) ./ wtmp[:, k].^2
                return w, H, ind
            end
        end
    end
    @warn "Not converged after $(maxiter) steps. Results may be inaccurate."
end

"""train + predict"""
function RVM!(
    XH::Array{T}, XL::Array{T}, t::Matrix{T}, XLtest::Matrix{T},
    α::Matrix{T}, β::Matrix{T},
    method::Symbol=:block; tol=1e-5, maxiter=100000, n_samples=5000
) where T<:Real
    # Multinomial
    n = size(X, 1)
    d = size(X, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))
    K = size(t, 2)  # total number of classes
    size(α, 2) == K || throw(DimensionMismatch("Number of classes and size of α mismatch."))

    wh, H, ind_h = RVM!(
        XH, t, α, tol=tol, maxiter=maxiter
    )
    ind_nonzero = findall(in(findall(x -> x > 1e-3, std(XL, dims=1)[:])), ind_h)
    ind = ind_h[ind_nonzero]
    n_ind = size(ind, 1)
    # initialise
    # preallocate type-II likelihood (evidence) vector
    wh_samples = Array{T}(undef, n_ind, K, n_samples)
    for k ∈ 1:K
        wh_samples[:, k, :] .= rand(
            MvNormal(
                view(wh, ind_nonzero, k),
                view(H, ind_nonzero, ind_nonzero, k)
            ), n_samples
        )
    end
    llh2 = Vector{T}(undef, maxiter)
    fill!(llh2, -Inf)
    βtmp = @view β[ind, :]
    XLtmp = @view XL[:, ind]
    XLtesttmp = @view XLtest[:, ind]
    #w = ones(T, d, K) * 0.00001
    #αp = ones(T, d, K)
    #A, Y, logY = (Matrix{T}(undef, n, K) for _ = 1:3)
    for iter ∈ 2:maxiter
        ind_l = unique!([item[1] for item in findall(αtmp .< 10000)])
        n_ind_l = size(ind_l, 1)
        #copyto!(αp, α)
        β2 = copy(βtmp[ind_l, :])
        @views g = eachslice(whsamples, dims=3) |>
        Map(
            x -> Logit(
                x, β2, XLtmp[:, ind_l],
                transpose(XLtmp[:, ind_l]),
                t, tol, maxiter
            )
        ) |> Broadcasting() |> Folds.sum
        g ./= n_samples
        # update β
        @views llh[iter] = sum(g[end, :])
        @views βtmp[ind_l, :] .=
            (1 .- βtmp[ind_l, :] .* g[(n_ind_l+1):(end-1), :]) ./ (g[1:n_ind_l, :]).^2
        # check convergence
        incr = abs((llh2[iter] - llh2[iter-1]) / llh2[iter-1])
        @info "iteration $iter" incr
        if incr < tol
            @views g = eachslice(whsamples, dims=3) |>
            Map(
                x -> Logit(
                    x, β2, XLtmp[:, ind_l],
                    transpose(XLtmp[:, ind_l]),
                    t, XLtesttmp[:, ind_l], tol, maxiter
                )
            ) |> Broadcasting() |> Folds.sum
            g ./= n_samples
            return g
        end
    end
    @warn "Not converged after $(maxiter) steps. Results may be inaccurate."
end

function Logit!(
    w::AbstractMatrix{T}, α::AbstractMatrix{T}, X::AbstractArray{T},
    t::AbstractMatrix{Int64}, tol::Float64, maxiter::Int64,
    A::AbstractMatrix{T}, Y::AbstractMatrix{T}, logY::AbstractMatrix{T}
) where T<:Real
    n = size(t, 1)
    d = size(X, 2)
    K = size(t, 2) # number of classes
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
        mul!(g, Xt, t .- Y)
        g .-= w .* α
        copyto!(wp, w)
        # update weights
        w .+= g .* r
        mul!(A, X, w)
        logY .= A .- log.(sum(exp.(A), dims=2))
        Y .= exp.(logY)
        # update likelihood
        llh = -0.5sum(α .* w .* w) + sum(t .* logY)
        while (llh - llhp < 0)  # line search
            g ./= 2
            w .= wp .+ g .* r
            mul!(A, X, w)
            logY .= A .- log.(sum(exp.(A), dims=2))
            Y .= exp.(logY)
            llh = -0.5sum(α .* w .* w) + sum(t .* logY)
        end
        if llh - llhp < tol
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

function Logit(
    wh::AbstractMatrix{T}, α::AbstractMatrix{T},
    X::AbstractArray{T}, Xt::AbstractArray{T},
    t::AbstractMatrix{T},
    tol::Float64, maxiter::Int64
) where T<:Real
    # need a sampler
    n, d = size(X)
    K = size(t, 2) # number of classes
    wp, g, gp = (similar(wh) for _ = 1:3)
    wl = copy(wh)
    A, Y, logY = (Matrix{T}(undef, n, K) for _ = 1:3)
    mul!(A, X, wh)
    logY .= A .- log.(sum(exp.(A), dims=2))
    Y .= exp.(logY)
    llhp = -Inf
    r = [0.00001]
    for iter = 2:maxiter
        # update gradient
        copyto!(gp, g)
        mul!(g, Xt, t .- y)
        g .-= α .* wl
        #ldiv!(factorize(H), g)
        # update w
        copyto!(wp, wl)
        wl .+= g .* r
        mul!(A, X, wl)
        logY .= A .- log.(sum(exp.(A), dims=2))
        llh = -0.5sum(α .* wl .* wl) + sum(t .* logY)
        while llh - llhp < 0.0
            g ./= 2
            wl .= wp .+ g .* r
            mul!(A, X, wl)
            logY .= A .- log.(sum(exp.(A), dims=2))
            llh = -0.5sum(α .* wl .* wl) + sum(t .* logY)
        end
        Y .= exp.(logY)
        if llh - llhp < tol
            for k ∈ 1:K
                @views WoodburyInv!(
                    g[:, k], α[:, k],
                    Diagonal(sqrt.(Y[:, k] .* (1 .- Y[:, k]))) * X
                )
                #predict!(y, Xtest, wl, H, 1:d)
            end
            return vcat(
                (wl.-wh).^2, g,
                -0.5sum(α .* wl .* wl, dims=1) .+ sum(t .* logY, dims=1)
            )#, llh+0.5logdet(H))
        else
            llhp = llh
            r .= abs(sum((wl .- wp) .* (g .- gp))) ./ sum((g .- gp) .^ 2)
        end
    end
    @warn "Not converged in finding the posterior of wh."
end

function Logit(
    wh::AbstractMatrix{T}, α::AbstractMatrix{T},
    X::AbstractArray{T}, Xt::AbstractArray{T},
    t::AbstractMatrix{T}, Xtest::AbstractArray{T},
    tol::Float64, maxiter::Int64
) where T<:Real
    # need a sampler
    n, d = size(X)
    K = size(t, 2) # number of classes
    wp, g, gp = (similar(wh) for _ = 1:3)
    wl = copy(wh)
    A, Y, logY = (Matrix{T}(undef, n, K) for _ = 1:3)
    mul!(A, X, wh)
    logY .= A .- log.(sum(exp.(A), dims=2))
    Y .= exp.(logY)
    llhp = -Inf
    r = [0.00001]
    for iter = 2:maxiter
        # update gradient
        copyto!(gp, g)
        mul!(g, Xt, t .- y)
        g .-= α .* wl
        #ldiv!(factorize(H), g)
        # update w
        copyto!(wp, wl)
        wl .+= g .* r
        mul!(A, X, wl)
        logY .= A .- log.(sum(exp.(A), dims=2))
        llh = -0.5sum(α .* wl .* wl) + sum(t .* logY)
        while llh - llhp < 0.0
            g ./= 2
            wl .= wp .+ g .* r
            mul!(A, X, wl)
            logY .= A .- log.(sum(exp.(A), dims=2))
            llh = -0.5sum(α .* wl .* wl) + sum(t .* logY)
        end
        Y .= exp.(logY)
        if llh - llhp < tol
            H = Array{T}(undef, d, d, K)
            for k ∈ 1:K
                @views H[:, :, k] .= WoodburyInv!(
                    α[:, k],
                    Diagonal(sqrt.(Y[:, k] .* (1 .- Y[:, k]))) * X
                )
                predict!(Y, Xtest, wl, H, 1:d)
            end
            return Y
        else
            llhp = llh
            r .= abs(sum((wl .- wp) .* (g .- gp))) ./ sum((g .- gp) .^ 2)
        end
    end
    @warn "Not converged in finding the posterior of wh."
end
