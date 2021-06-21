# multiclass
function predict(
    X::AbstractMatrix{T}, w::AbstractMatrix{T},
    H::AbstractArray{T, 3}, ind::AbstractVector{Int64}
) where T <: Real
    Xview = @view X[:, ind]
    K = size(w, 2)
    n = size(X, 1)
    A = Matrix{T}(undef, n, K)
    Xt = transpose(Xview)
    @inbounds for k ∈ 1:K
        #p = view(A, :, k)
        # p .= diag(X * view(H, :, :, k) * Xt)
        A[:, k] .= (
            1 .+ π .* diag(Xview * view(H, :, :, k) * Xt) ./ 8
        ).^(-0.5) .* (Xview * view(w, :, k))
    end
    return A
end

function predict!(
    A::AbstractMatrix{T}, X::AbstractMatrix{T},
    w::AbstractMatrix{T}, H::AbstractArray{T, 3}#, ind::AbstractVector{Int64}
) where T <: Real
    #Xview = @view X[:, ind]
    n, d = size(X)
    K = size(w, 2)
    #@inbounds for k ∈ 1:K
    #    # p = view(A, :, k)
    #    # p .= diag(X * view(H, :, :, k) * Xt)
    #    A[:, k] .= (
    #        1 .+ π .* diag(X * view(H, :, :, k) * Xt) ./ 8
    #    ).^(-0.5) .* (X * view(w, :, k))
    #end
    ### using LoopVectorization
    fill!(A, 1.)
    LoopVectorization.@turbo for k ∈ 1:K
        for nn ∈ 1:n, i ∈ 1:d, j ∈ 1:d
            A[nn, k] += (π / 8) * X[nn, i] * H[i, j, k] * X[nn, j]
        end
    end
    A .= A.^(-0.5)
    A .*= X * w
    LoopVectorization.@avx A .= exp.(A) ./ sum(exp.(A), dims=2)
    return A
end

function RVM!(
    X::AbstractMatrix{T}, t::AbstractMatrix{T}, α::AbstractMatrix{T};
    rtol=1e-5, atol=1e-8, maxiter=100000
) where T<:Real
    # Multinomial
    n = size(X, 1)
    d = size(X, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))
    K = size(t, 2)  # total number of classes
    size(α, 2) == K || throw(DimensionMismatch("Number of classes and size of α mismatch."))
    ind_nonzero = findall(x -> x > 1e-3, std(X, dims=1)[:])
    # initialise
    # preallocate type-II likelihood (evidence) vector
    llh2 = Vector{T}(undef, maxiter)
    fill!(llh2, -Inf)
    w = ones(T, d, K) .* 0.00001
    #αp = ones(T, d, K)
    A, Y, logY = (Matrix{T}(undef, n, K) for _ = 1:3)
    r = [0.0001]
    prog = ProgressUnknown(
        "training on high quality data...",
        spinner=true
    )
    for iter ∈ 2:maxiter
        ind_h = unique!([item[1] for item in findall(α .< 10000)])
        ind = ind_h[findall(in(ind_nonzero), ind_h)]
        n_ind = size(ind, 1)
        αtmp = copy(α[ind, :])
        wtmp = copy(w[ind, :])
        Xtmp = copy(X[:, ind])
        #copyto!(αp, α)
        llh2[iter] = Logit!(
            wtmp, αtmp, Xtmp,
            t, atol, maxiter, A, Y, logY, r
        )
        w[ind, :] .= wtmp
        # update α
        @inbounds Threads.@threads for k ∈ 1:K
            # update alpha - what is y?
            #@views mul!(a, X, wtmp[:, k])
            #y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
            α2 = view(αtmp, :, k)
            yk = view(Y, :, k)
            yk .= yk .* (1 .- yk)
            yk[yk .< 1e-10] .= 0.
            WoodburyInv!(
                α2, α[ind, k],
                Diagonal(sqrt.(yk)) * Xtmp
            )
            α[ind, k] .= (1 .- α[ind, k] .* α2) ./ view(wtmp, :, k).^2
        end
        #@info "α" α[ind, :]
        # check convergence
        incr = abs((llh2[iter] - llh2[iter-1]) / llh2[iter-1])
        #@info "iteration $iter" incr
        ProgressMeter.next!(
            prog;
            showvalues = [(:iter,iter-1), (:incr,incr)]
        )
        if incr < rtol
            ProgressMeter.finish!(prog, spinner = '✓')
            H = Array{T}(undef, n_ind, n_ind, K)
            @inbounds Threads.@threads for k ∈ 1:K
                yk = view(Y, :, k)
                yk .= yk .* (1 .- yk)
                yk[yk .< 1e-10] .= 0.
                H[:, :, k] .= WoodburyInv!(
                    α[ind, k],
                    Diagonal(sqrt.(yk)) * Xtmp
                )
            end
            return wtmp, H, ind
        end
    end
    ProgressMeter.finish!(prog, spinner = '✗')
    @warn "Not converged after $(maxiter) steps. Results may be inaccurate."
end

"""train + predict"""
function RVM!(
    XH::AbstractMatrix{T}, XL::AbstractMatrix{T}, t::AbstractMatrix{T},
    XLtest::AbstractMatrix{T}, α::AbstractMatrix{T}, β::AbstractMatrix{T};
    rtol=1e-5, atol=1e-7, maxiter=100000, n_samples=2000
) where T<:Real
    # Multinomial
    n = size(XL, 1)
    d = size(XL, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))
    K = size(t, 2)  # total number of classes
    size(α, 2) == K || throw(DimensionMismatch("Number of classes and size of α mismatch."))

    wh, H, ind_h = RVM!(
        XH, t, α, rtol=rtol, maxiter=maxiter
    )
    ind_nonzero = findall(in(findall(x -> x > 1e-3, std(XL, dims=1)[:])), ind_h)
    ind = ind_h[ind_nonzero]
    n_ind = size(ind, 1)
    # initialise
    # preallocate type-II likelihood (evidence) vector
    llh2 = Vector{T}(undef, maxiter)
    fill!(llh2, -Inf)
    # posterior of wh
    whsamples = Array{T}(undef, n_samples, n_ind * K)
    Threads.@threads for k ∈ 1:K
        whsamples[:, ((k-1)*n_ind + 1):(k*n_ind)] .= transpose(rand(
            MvNormal(
                wh[ind_nonzero, k],
                H[ind_nonzero, ind_nonzero, k]
            ), n_samples
        ))
    end
    whsamples = reshape(transpose(whsamples), n_ind, K, n_samples)
    # screening
    βtmp = @view β[ind, :]
    XLtmp = @view XL[:, ind]
    XLtesttmp = @view XLtest[:, ind]
    #w = ones(T, d, K) * 0.00001
    #αp = ones(T, d, K)
    #A, Y, logY = (Matrix{T}(undef, n, K) for _ = 1:3)
    prog = ProgressUnknown(
        "training on low quality data...",
        spinner=true
    )
    for iter ∈ 2:maxiter
        ind_l = unique!([item[1] for item in findall(βtmp .< 10000)])
        #@info "ind_l" ind_l
        n_ind_l = size(ind_l, 1)
        #copyto!(αp, α)
        β2 = copy(βtmp[ind_l, :])
        @info "β2" β2
        XL2 = copy(XLtmp[:, ind_l])
        g = eachslice(whsamples, dims=3) |>
        Map(
            x -> Logit(
                x[ind_l, :], β2, XL2,
                transpose(XL2),
                t, atol, maxiter
            )
        ) |> Broadcasting() |> Folds.sum
        g ./= n_samples
        @info "g" g
        # update β
        βtmp[ind_l, :] .=
            @views (1 .- β2 .* g[(n_ind_l+1):(end-1), :]) ./ (g[1:n_ind_l, :].^2 .+ 1e-6)
        # check convergence
        llh2[iter] = sum(g[end, :])
        incr = abs((llh2[iter] - llh2[iter-1]) / llh2[iter-1])
        #@info "iteration $iter" incr
        ProgressMeter.next!(
            prog;
            showvalues = [(:iter,iter-1), (:incr,incr)]
        )
        if incr < rtol
            ProgressMeter.finish!(prog, spinner = '✓')
            XLtest2 = copy(XLtesttmp[:, ind_l])
            #XLtest2t = transpose(XLtest2t)
            g = eachslice(whsamples, dims=3) |>
            Map(
                x -> Logit(
                    x[ind_l, :], β2, XL2,
                    transpose(XL2),
                    t, XLtest2, tol, maxiter
                )
            ) |> Broadcasting() |> Folds.sum
            g ./= n_samples
            return g
        end
    end
    ProgressMeter.finish!(prog, spinner = '✗')
    @warn "Not converged after $(maxiter) steps. Results may be inaccurate."
end

function Logit!(
    w::AbstractMatrix{T}, α::AbstractMatrix{T}, X::AbstractMatrix{T},
    t::AbstractMatrix{T}, tol::Float64, maxiter::Int64,
    A::AbstractMatrix{T}, Y::AbstractMatrix{T}, logY::AbstractMatrix{T},
    r::AbstractArray{T}
) where T<:Real
    n = size(t, 1)
    d = size(X, 2)
    K = size(t, 2) # number of classes
    Xt = transpose(X)
    ind = findall(x -> x < 10000, α[:])
    #dk = d * Ks
    g, wp, gp = (zeros(d, K) for _ = 1:3)
    llhp = -Inf
    mul!(A, X, w)
    LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
    LoopVectorization.@avx Y .= exp.(logY)
    #r = 1  # initial step size
    for iter = 2:maxiter
        # update gradient
        mul!(g, Xt, t .- Y)
        g .-= w .* α
        #@info "g" g[ind]
        copyto!(wp, w)
        # update weights
        w[ind] .+= @views g[ind] .* r
        #@info "w" findall(isnan, w)
        #w .+= g .* r
        mul!(A, X, w)
        LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
        # update likelihood
        #llh = @views -0.5sum(α[ind] .* w[ind] .* w[ind]) + sum(t .* logY)
        llh = @views -0.5sum(α[ind] .* w[ind] .* w[ind]) + sum(t .* logY)
        #@info "llh" llh
        #@info "r * sum(g.^2) / 2" r * sum(g.^2) / 2
        while !(llh - llhp > 0) #r[1] * sum(g[ind].^2) / 2 # line search
            #g ./= 2
            r .*= 0.8
            w[ind] .= @views wp[ind] .+ g[ind] .* r
            #w .= wp .+ g .* r
            mul!(A, X, w)
            LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
            llh = @views -0.5sum(α[ind] .* w[ind] .* w[ind]) + sum(t .* logY)
            #llh = -0.5sum(α .* w .* w) + sum(t .* logY)
        end
        #@info "w" w
        #@info "r" r
        LoopVectorization.@avx Y .= exp.(logY)
        #@info "Y" Y
        #@info "incr" llh - llhp
        if llh - llhp < tol
            return llh
        else
            llhp = llh
            # update step sizeß
            r .= @views abs(sum((w[ind] .- wp[ind]) .* (g[ind] .- gp[ind]))) / sum((g[ind] .- gp[ind]) .^ 2)
            #@info "r" r
            #r .= 0.00001
            copyto!(gp, g)
        end
    end
    @warn "not converged."
end

function Logit(
    wh::AbstractMatrix{T}, α::AbstractMatrix{T},
    X::AbstractMatrix{T}, Xt::AbstractMatrix{T},
    t::AbstractMatrix{T}, tol::Float64, maxiter::Int64
) where T<:Real
    # need a sampler
    n, d = size(X)
    K = size(t, 2) # number of classes
    wp, g, gp = (similar(wh) for _ = 1:3)
    wl = copy(wh)
    A, Y, logY = (Matrix{T}(undef, n, K) for _ = 1:3)
    mul!(A, X, wl)
    LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
    LoopVectorization.@avx Y .= exp.(logY)
    llhp = -Inf
    r = [0.0001]
    ind = findall(x -> x < 10000, α[:])
    for iter = 2:maxiter
        # update gradient
        mul!(g, Xt, t .- Y)
        g[ind] .-= @views α[ind] .* wl[ind]
        #ldiv!(factorize(H), g)
        # update w
        copyto!(wp, wl)
        wl[ind] .+= @views g[ind] .* r
        mul!(A, X, wl)
        LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
        llh = @views -0.5sum(α[ind] .* wl[ind] .* wl[ind]) + sum(t .* logY)
        while !(llh - llhp > 0)
            r .*= 0.8
            wl[ind] .= @views wp[ind] .+ g[ind] .* r
            mul!(A, X, wl)
            LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
            llh = @views -0.5sum(α[ind] .* wl[ind] .* wl[ind]) + sum(t .* logY)
        end
        LoopVectorization.@avx Y .= exp.(logY)
        if llh - llhp < tol
            @inbounds for k ∈ 1:K
                yk = view(Y, :, k)
                yk .= yk .* (1 .- yk)
                yk[yk .< 1e-10] .= 0.
                gk = view(g, :, k)
                αk = view(α, :, k)
                WoodburyInv!(
                    gk, αk,
                    Diagonal(sqrt.(yk)) * X
                )
            end
            @views return vcat(
                (wl.-wh).^2, g,
                -0.5sum(α[ind] .* wl[ind] .* wl[ind], dims=1) .+
                sum(t .* logY, dims=1)
            )#, llh+0.5logdet(H))
        else
            llhp = llh
            r .= @views abs(sum((wl[ind] .- wp[ind]) .* (g[ind] .- gp[ind]))) / (sum((g[ind] .- gp[ind]) .^ 2) + 1e-3)
            copyto!(gp, g)
        end
    end
    @warn "Not converged."
end

function Logit(
    wh::AbstractMatrix{T}, α::AbstractMatrix{T},
    X::AbstractMatrix{T}, Xt::AbstractMatrix{T},
    t::AbstractMatrix{T}, Xtest::AbstractMatrix{T},
    tol::Float64, maxiter::Int64
) where T<:Real
    # need a sampler
    n, d = size(X)
    K = size(t, 2) # number of classes
    wp, g, gp = (similar(wh) for _ = 1:3)
    wl = copy(wh)
    A, Y, logY = (Matrix{T}(undef, n, K) for _ = 1:3)
    mul!(A, X, wh)
    LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
    LoopVectorization.@avx Y .= exp.(logY)
    llhp = -Inf
    r = [0.00001]
    ind = findall(x -> x < 10000, α[:])
    for iter = 2:maxiter
        # update gradient
        mul!(g, Xt, t .- Y)
        g[ind] .-= α[ind] .* wl[ind]
        copyto!(wp, wl)
        wl[ind] .+= @views g[ind] .* r
        mul!(A, X, wl)
        LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
        llh = @views -0.5sum(α[ind] .* wl[ind] .* wl[ind]) + sum(t .* logY)
        while !(llh - llhp > 0)
            r .*= 0.8
            wl[ind] .= @views wp[ind] .+ g[ind] .* r
            mul!(A, X, wl)
            LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
            llh = @views -0.5sum(α[ind] .* wl[ind] .* wl[ind]) + sum(t .* logY)
        end
        LoopVectorization.@avx Y .= exp.(logY)
        if llh - llhp < tol
            H = Array{T}(undef, d, d, K)
            @inbounds for k ∈ 1:K
                yk = view(Y, :, k)
                yk .= yk .* (1 .- yk)
                yk[yk .< 1e-10] .= 0.
                αk = view(α, :, k)
                H[:, :, k] .= WoodburyInv!(
                    αk,
                    Diagonal(sqrt.(yk)) * X
                )
                pred = Matrix{T}(undef, size(Xtest, 1), K)
                predict!(pred, Xtest, wl, H, 1:d)
            end
            return pred
        else
            llhp = llh
            r .= @views abs(sum((wl[ind] .- wp[ind]) .* (g[ind] .- gp[ind]))) / sum((g[ind] .- gp[ind]) .^ 2)
            copyto!(gp, g)
        end
    end
    @warn "Not converged."
end
