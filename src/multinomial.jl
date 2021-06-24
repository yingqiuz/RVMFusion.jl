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
    LoopVectorization.@avx A .= exp.(A) ./ sum(exp.(A), dims=2)
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
        ind_h = unique!([item[1] for item in findall(α .< (1/rtol))])
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
        #LoopVectorization.@avx llh2[iter] += 0.5sum(log.(αtmp))
        #llh2[iter] -= 0.5 * n_ind * log(2π)
        w[ind, :] .= wtmp
        # update α
        @inbounds Threads.@threads for k ∈ 1:K
            # update alpha - what is y?
            #@views mul!(a, X, wtmp[:, k])
            #y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
            αk = view(αtmp, :, k)
            yk = view(Y, :, k)
            yk .= yk .* (1 .- yk)
            yk[yk .< 1e-10] .= 0.
            WoodburyInv!(
                αk, α[ind, k],
                Diagonal(sqrt.(yk)) * Xtmp
            )
            α[ind, k] .= (1 .- α[ind, k] .* αk) ./ view(wtmp, :, k).^2
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
    rtol=1e-6, atol=1e-8, maxiter=100000, n_samples=2000
) where T<:Real
    # Multinomial
    n = size(XL, 1)
    d = size(XL, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))
    K = size(t, 2)  # total number of classes
    size(α, 2) == K || throw(DimensionMismatch("Number of classes and size of α mismatch."))

    wh, H, ind_h = RVM!(
        XH, t, α, rtol=rtol, atol=atol, maxiter=maxiter
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
    wh =  wh[ind_nonzero, :]
    H = H[ind_nonzero, ind_nonzero, :]
    Threads.@threads for k ∈ 1:K
        whsamples[:, ((k-1)*n_ind + 1):(k*n_ind)] .= transpose(rand(
            MvNormal(
                wh[:, k],
                H[:, :, k]
            ), n_samples
        ))
    end
    whsamples = reshape(transpose(whsamples), n_ind, K, n_samples)
    # screening
    βtmp = @view β[ind, :]
    XLtmp = @view XL[:, ind]
    XLtesttmp = @view XLtest[:, ind]
    #@info "whsamples" size(whsamples)
    #w = ones(T, d, K) * 0.00001
    #αp = ones(T, d, K)
    #A, Y, logY = (Matrix{T}(undef, n, K) for _ = 1:3)
    prog = ProgressUnknown(
        "training on low quality data...",
        spinner=true
    )
    for iter ∈ 2:maxiter
        ind_l = unique!([item[1] for item in findall(βtmp .< (1/rtol))])
        #@info "ind_l" ind_l
        n_ind_l = size(ind_l, 1)
        if n_ind_l == 0
            ProgressMeter.finish!(prog, spinner = '✓')
            return predict(XLtesttmp, wh, H, 1:n_ind)
        end
        #copyto!(αp, α)
        β2 = copy(βtmp[ind_l, :])
        XL2 = copy(XLtmp[:, ind_l])
        non_inf_ind = findall(x->x<(1/rtol), β2[:])
        n_non_inf_ind = size(non_inf_ind, 1)
        #@info "n_non_inf_ind" n_non_inf_ind
        #g = zeros(T, 2n_non_inf_ind + 1)
        #for nn ∈ 1:n_samples
        #    g .+= Logit(whsamples[ind_l, :, nn], β2, XL2, transpose(XL2), t, non_inf_ind, atol, maxiter)
        #end
        g = eachslice(whsamples, dims=3) |>
        Map(
            x -> Logit(
                x[ind_l, :], β2, XL2,
                transpose(XL2),
                t, non_inf_ind, atol, maxiter
            )
        ) |> Broadcasting() |> Folds.sum
        g ./= n_samples
        #@info "g" g
        # update β
        #@info "β2" β2[non_inf_ind]
        llh2[iter] = g[end] - 0.5 * n_non_inf_ind * log(2π)
        llh2[iter] += LoopVectorization.@avx 0.5sum(log.(view(β2, non_inf_ind)))
        β2[non_inf_ind] .=
            (1 .- β2[non_inf_ind] .* view(g, 1+n_non_inf_ind:2n_non_inf_ind)) ./
            view(g, 1:n_non_inf_ind)
        βtmp[ind_l, :] .= β2
        #@info "β2" β2[non_inf_ind]
        #@info "wl.^2" g[1:n_non_inf_ind]
        #βtmp[ind_l, :] .= @views (1 .- β2.*g[n_ind_l+1:2n_ind_l, :]) ./ g[1:n_ind_l, :]
        # check convergence
        incr = abs((llh2[iter] - llh2[iter-1]) / llh2[iter-1])
        #@info "iter $(iter) incr" incr
        #@info "iteration $iter" incr
        ProgressMeter.next!(
            prog;
            showvalues = [(:iter,iter-1), (:incr,incr)]
        )
        #@info "incr" incr
        #@info "llh" llh2[iter]
        #@info "βtmp" βtmp
        if incr < rtol
            ProgressMeter.finish!(prog, spinner = '✓')
            XLtest2 = copy(XLtesttmp[:, ind_l])
            #XLtest2t = transpose(XLtest2t)
            g = eachslice(whsamples, dims=3) |>
            Map(
                x -> Logit(
                    x[ind_l, :], β2, XL2,
                    transpose(XL2),
                    t, XLtest2, non_inf_ind, atol, maxiter
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
            return llh #+ 0.5sum(log.(α[ind]))
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
    return -Inf
end

function Logit(
    wh::AbstractMatrix{T}, α::AbstractMatrix{T},
    X::AbstractMatrix{T}, Xt::AbstractMatrix{T},
    t::AbstractMatrix{T}, ind::AbstractArray{Int64},
    tol::Float64, maxiter::Int64
) where T<:Real
    # need a sampler
    n, d = size(X)
    K = size(t, 2) # number of classes
    wp, wl, g, gp = (zeros(T, d, K) for _ = 1:4)
    A, Y, logY = (Matrix{T}(undef, n, K) for _ = 1:3)
    mul!(A, X, wl .+ wh)
    LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
    LoopVectorization.@avx Y .= exp.(logY)
    llhp = -Inf
    r = [0.0001]
    #ind = findall(x -> x < 10000, α[:])
    #@info "wh" size(wh)
    #@info "llhp" llhp
    for iter = 2:maxiter
        #if iter % 100 == 0
            #println(iter)
        #end
        # update gradient
        mul!(g, Xt, t .- Y)
        g .-= α .* wl
        #ldiv!(factorize(H), g)
        # update w
        copyto!(wp, wl)
        wl[ind] .+= @views g[ind] .* r
        mul!(A, X, wl .+ wh)
        LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
        llh = -0.5sum(α[ind] .* (wl[ind]).^ 2) + sum(t .* logY) #+ 0.5sum(log.(α))
        while !(llh - llhp > 0)
            r .*= 0.8
            wl .= wp .+ g .* r
            mul!(A, X, wl .+ wh)
            LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
            llh = -0.5sum(α[ind] .* (wl[ind]).^ 2) + sum(t .* logY) #+ 0.5sum(log.(α))
        end
        LoopVectorization.@avx Y .= exp.(logY)
        #@info "llh - llhp" llh - llhp
        if llh - llhp < tol
            #@info "llh" llh
            #@info "g" g
            #@info "Y" Y
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
            #@info "g" g

            #return vcat(
            #    (wl).^2, g,
            #    -0.5sum(α .* wl .* wl, dims=1) .+
            #    sum(t .* logY, dims=1)
            #)#, llh+0.5logdet(H))
            return vcat(
                wl[ind].^2,
                g[ind],
                llh
            )
        else
            llhp = llh
            r .= abs(sum((wl[ind] .- wp[ind]) .* (g[ind] .- gp[ind]))) / sum((g[ind] .- gp[ind]) .^ 2)
            copyto!(gp, g)
        end
    end
    @warn "Not converged."
    return zeros(T, size(ind, 1))
end

function Logit(
    wh::AbstractMatrix{T}, α::AbstractMatrix{T},
    X::AbstractMatrix{T}, Xt::AbstractMatrix{T},
    t::AbstractMatrix{T}, Xtest::AbstractMatrix{T},
    ind::AbstractArray{Int64}, tol::Float64, maxiter::Int64
) where T<:Real
    # need a sampler
    n, d = size(X)
    K = size(t, 2) # number of classes
    wp, wl, g, gp = (zeros(T, d, K) for _ = 1:4)
    A, Y, logY = (Matrix{T}(undef, n, K) for _ = 1:3)
    mul!(A, X, wh .+ wl)
    LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
    LoopVectorization.@avx Y .= exp.(logY)
    llhp = -Inf
    r = [0.00001]
    #ind = findall(x -> x < 10000, α[:])
    for iter = 2:maxiter
        #if iter % 100 == 0
            #println(iter)
        #end
        # update gradient
        mul!(g, Xt, t .- Y)
        g .-= α .* wl
        copyto!(wp, wl)
        wl[ind] .+= @views g[ind] .* r
        mul!(A, X, wl .+ wh)
        LoopVectorization.@avx logY .= A .- log.(sum(exp.(A), dims=2))
        llh = @views -0.5sum(α[ind] .* wl[ind] .* wl[ind]) + sum(t .* logY)
        while !(llh - llhp > 0)
            r .*= 0.8
            wl[ind] .= @views wp[ind] .+ g[ind] .* r
            mul!(A, X, wl .+ wh)
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
            end
            pred = Matrix{T}(undef, size(Xtest, 1), K)
            predict!(pred, Xtest, wl .+ wh, H)
            #pred = predict(Xtest, wl .+ wh, H, 1:d)
            return pred
        else
            llhp = llh
            r .= @views abs(sum((wl[ind] .- wp[ind]) .* (g[ind] .- gp[ind]))) / sum((g[ind] .- gp[ind]) .^ 2)
            copyto!(gp, g)
        end
    end
    @warn "Not converged."
end
