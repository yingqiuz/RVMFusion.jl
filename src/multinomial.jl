# multiclass
# define types
struct MnRVModel{T<:Real} <: Model
    w::AbstractArray{T, 2}
    H::AbstractArray{T, 3}
    ind::AbstractArray{Int64, 1}
end

struct MnFusedRVModel{T<:Real} <: Model
    w::AbstractArray{T, 3}
    H::AbstractArray{T, 4}
    ind::AbstractArray{Int64, 1}
end

function predict(
    model::MnRVModel{T}, X::AbstractMatrix{T}
) where T <: Real
    K = size(model.w, 2)
    n = size(X, 1)
    Xview = @view X[:, model.ind]
    A = Matrix{T}(undef, n, K)
    Xt = transpose(Xview)
    Threads.@threads for k ∈ 1:K
        @inbounds A[:, k] .= (
            1 .+ π .* diag(Xview * view(H, :, :, k) * Xt) ./ 8
        ).^(-0.5) .* (Xview * view(w, :, k))
    end
    @avx A .= exp.(A) ./ sum(exp.(A), dims=2)
    return A
end

function predict(
    model::MnFusedRVModel{T}, X::AbstractMatrix{T}
) where T <: Real
    Xnew = copy(X[:, model.ind])
    n = size(Xnew)
    d, K, n_samples = size(w)
    A = 1:n_samples |> Map(
        k -> predict(Xnew, model.w[:, :, k], model.H[:, :, :, k])
    ) |> Broadcasting() |> Folds.sum
    A ./= n_samples
    return A
end

function predict(
    X::AbstractMatrix{T},
    w::AbstractMatrix{T},
    H::AbstractArray{T, 3}
) where T <: Real
    n, d = size(X)
    K = size(w, 2)
    p = Matrix{T}(undef, n, K)
    fill!(p, 1.)
    @turbo for k ∈ 1:K, nn ∈ 1:n, i ∈ 1:d, j ∈ 1:d
        p[nn, k] += (π / 8) * X[nn, i] * H[i, j, k] * X[nn, j]
    end
    p .= p.^(-0.5)
    p .*= X * w
    @avx p .= exp.(p)
    return p ./ sum(p, dims=2)
end

"""
model for higher quality data
"""
function RVM!(
    X::AbstractMatrix{T}, t::AbstractMatrix{T}, α::AbstractMatrix{T};
    rtol=1e-6, atol=1e-8, maxiter=50000, BatchSize=size(X, 1)
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
    llh = zeros(T, maxiter)
    llh[1] = -Inf
    w = zeros(T, d, K)
    #αp = ones(T, d, K)
    num_batches = convert(Int64, round(n / BatchSize))
    A1, Y1, logY1 = (Matrix{T}(undef, BatchSize, K) for _ = 1:3)
    A2, Y2, logY2 = (Matrix{T}(undef, n - BatchSize*(num_batches-1), K) for _ = 1:3)
    println("Setup done.")
    for iter ∈ 2:maxiter
        ind = unique!([item[1] for item in findall(α .< (1/rtol))])
        ind = ind[findall(in(ind_nonzero), ind)]
        n_ind = size(ind, 1)
        αtmp = copy(α[ind, :])
        wtmp = copy(w[ind, :])
        wp = copy(wtmp)
        g, gp = (zeros(T, n_ind, K) for _ = 1:2)
        η = [0.0001] # initial step size
        #ind_flat = findall(x -> x < (1/atol), αtmp[:])
        @showprogress 0.5 "epoch $(iter)" for b ∈ 1:num_batches
            if b != num_batches
                Xtmp = copy(X[(b-1)*BatchSize+1:b*BatchSize, ind])
                ttmp = copy(t[(b-1)*BatchSize+1:b*BatchSize, :])
                llh[iter] += Logit!(
                    wtmp, αtmp, Xtmp,
                    ttmp, atol, maxiter,
                    A1, Y1, logY1, η, g, gp, wp#, ind_flat
                ) / num_batches
            else  # the last batch
                Xtmp = copy(X[(b-1)*BatchSize+1:end, ind])
                ttmp = copy(t[(b-1)*BatchSize+1:end, :])
                llh[iter] += Logit!(
                    wtmp, αtmp, Xtmp,
                    ttmp, atol, maxiter,
                    A2, Y2, logY2, η, g, gp, wp#, ind_flat
                ) / num_batches
            end
        end
        w[ind, :] .= wtmp
        α[ind, :] .= αtmp
        # check convergence
        incr = (llh[iter] - llh[iter-1]) / llh[iter-1]
        println("epoch ", iter, " done. incr :", incr)
        if abs(incr) < rtol || iter == maxiter
            if iter == maxiter
                @warn "Not converged after $(maxiter) steps.
                    Results may be inaccurate."
            end
            H = zeros(T, n_ind, n_ind, K)
            for b ∈ 1:num_batches
                if b != num_batches
                    Xtmp = @view X[(b-1)*BatchSize+1:b*BatchSize, ind]
                    ttmp = @view t[(b-1)*BatchSize+1:b*BatchSize, :]
                    hessian!(H, Xtmp, αtmp, wtmp, A1, Y1)
                else  # the last batch
                    Xtmp = @view X[(b-1)*BatchSize+1:end, ind]
                    ttmp = @view t[(b-1)*BatchSize+1:end, :]
                    hessian!(H, Xtmp, αtmp, wtmp, A2, Y2)
                end
            end
            print("done.")
            return MnRVModel(wtmp, H ./ num_batches, ind)
        end
    end
end

"""
average InvH across batches
"""
function hessian!(
    H::AbstractArray{T, 3}, Xtmp::AbstractMatrix{T},
    αtmp::AbstractMatrix{T}, wtmp::AbstractMatrix{T},
    A::AbstractMatrix{T}, Y::AbstractMatrix{T},
) where T <: Float64
    mul!(A, Xtmp, wtmp)
    @avx Y .= exp.(A)
    Y ./= sum(Y, dims=2)
    @inbounds Threads.@threads for k ∈ 1:size(H, 3)
        yk = view(Y, :, k)
        yk .= yk .* (1 .- yk)
        yk[yk .< 1e-10] .= 0.
        H[:, :, k] .+= WoodburyInv!(
            view(αtmp, :, k),
            Diagonal(sqrt.(yk)) * Xtmp
        )
    end
end

"""
function to find mode of w, for higher quality data
"""
function Logit!(
    w::AbstractMatrix{T}, α::AbstractMatrix{T}, X::AbstractMatrix{T},
    t::AbstractMatrix{T}, tol::Float64, maxiter::Int64,
    A::AbstractMatrix{T}, Y::AbstractMatrix{T}, logY::AbstractMatrix{T},
    η::AbstractArray{T}, g::AbstractMatrix{T}, gp::AbstractMatrix{T},
    wp::AbstractMatrix{T}#, ind::AbstractVector{Int64}
) where T<:Real
    n, K = size(t)
    d = size(X, 2)
    Xt = transpose(X)
    llhp = -Inf
    mul!(A, X, w)
    @avx logY .= A .- log.(sum(exp.(A), dims=2))
    @avx Y .= exp.(logY)
    for iter = 2:maxiter
        # update gradient
        mul!(g, Xt, t .- Y)
        #g[ind] .-= @views w[ind] .* α[ind]
        g .-= w .* α
        copyto!(wp, w)
        #w[ind] .+= @views g[ind] .* η
        w .+= g .* η
        mul!(A, X, w)
        @avx logY .= A .- log.(sum(exp.(A), dims=2))
        # update likelihood
        #llh = @views -0.5sum(α[ind] .* w[ind] .* w[ind]) + sum(t .* logY)
        llh = -0.5sum(α .* w .* w) + sum(t .* logY)
        while !(llh - llhp > 0) # line search
            η .*= 0.8
            w .= wp .+ g .* η
            mul!(A, X, w)
            @avx logY .= A .- log.(sum(exp.(A), dims=2))
            llh = -0.5sum(α .* w .* w) + sum(t .* logY)
        end
        @avx Y .= exp.(logY)
        if llh - llhp < tol || iter == maxiter
            if iter == maxiter
                @warn "not converged."
            end
            llh += 0.5sum(log.(view(α, ind))) - 0.5*size(ind, 1)*log(2π)
            # update α
            @inbounds Threads.@threads for k ∈ 1:K
                αk = view(α, :, k)
                gk = view(g, :, k)
                yk = view(Y, :, k)
                yk .= yk .* (1 .- yk)
                yk[yk .< 1e-10] .= 0.
                WoodburyInv!(
                    gk, αk,
                    Diagonal(sqrt.(yk)) * X
                )
                αk .= (1 .- αk .* gk) ./ (view(w, :, k) .^ 2 .+ 1e-10)
            end
            return llh
        else
            llhp = llh
            # update step size
            η .= @views abs(
                sum((w[ind] .- wp[ind]) .* (g[ind] .- gp[ind]))
            ) / sum((g[ind] .- gp[ind]) .^ 2)
            copyto!(gp, g)
        end
    end
end

"""
set up function for low quality data
"""
function setup(
    model::MnRVModel{T}, XL::AbstractMatrix{T},
    β::AbstractMatrix{T}, t::AbstractMatrix{T}, BatchSize::Int64
) where T<:Real
    n, d = size(XL)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(β, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))
    K = size(t, 2)  # total number of classes
    size(β, 2) == K || throw(DimensionMismatch("Number of classes and size of α mismatch."))
    num_batches = convert(Int64, round(n / BatchSize))
    # pruning
    ind_nonzero = findall(
        in(findall(x -> x > 1e-3, std(XL, dims=1)[:])),
        model.ind
    )
    ind = model.ind[ind_nonzero]
    n_ind = size(ind, 1)
    # preallocate type-II likelihood (evidence) vector
    llh = zeros(T, maxiter)
    llh[1] = -Inf
    # posterior of wh
    whsamples = Array{T}(undef, n_samples, n_ind * K)
    Threads.@threads for k ∈ 1:K
        whsamples[:, ((k-1)*n_ind + 1):(k*n_ind)] .= transpose(rand(
            MvNormal(
                model.w[ind_nonzero, k],
                model.H[ind_nonzero, ind_nonzero, k]
            ), n_samples
        ))
    end
    whsamples = reshape(transpose(whsamples), n_ind, K, n_samples)
    # screening
    wl = ones(T, n_ind, K) .* 1e-10
    prog = ProgressUnknown(
        "training on low quality data...",
        spinner=true
    )
    return prog, wl, whsamples, llh, num_batches, ind, n_ind, n, d, K
end

"""
iterate through all batches
"""
function epoch!(
    β::AbstractMatrix{T}, wl::AbstractMatrix{T},
    XL::AbstractMatrix{T}, t::AbstractMatrix{T},
    whsamples::AbstractArray{T, 3}, ind_l::AbstractVector{Int64},
    llh::AbstractVector{Float64},
    rtol::Float64, atol::Float64, maxiter::Int64, BatchSize::Int64,
    num_batches::Int64
) where T<:Real
    # flat index
    βtmp = copy(β[ind_l, :])
    wltmp = copy(wl[ind_l, :])
    non_inf_ind = findall(x->x<(1/rtol), βtmp[:])
    n_non_inf_ind = size(non_inf_ind, 1)
    for b ∈ 1:num_batches
        if b != num_batches
            XLtmp = copy(XL[(b-1)*BatchSize+1:b*BatchSize, ind_l])
            ttmp = copy(t[(b-1)*BatchSize+1:b*BatchSize, :])
        else  # the last batch
            XLtmp = copy(XL[(b-1)*BatchSize+1:end, ind_l])
            ttmp = copy(t[(b-1)*BatchSize+1:end, :])
        end
        g = eachslice(whsamples, dims=3) |>
        Map(
            x -> Logit(
                x[ind_l, :], wltmp, βtmp, XLtmp,
                transpose(XLtmp),
                ttmp, non_inf_ind, atol, maxiter
            )
        ) |> Broadcasting() |> Folds.sum
        g ./= n_samples
        llh[iter] += g[end] +
            @views 0.5sum(log.(βtmp[non_inf_ind])) -
            0.5n_non_inf_ind * log(2π)
        # update βtmp
        βtmp[non_inf_ind] .= @views (
            1 .- βtmp[non_inf_ind] .*
            g[n_non_inf_ind+1:2n_non_inf_ind]
        ) ./ g[1:n_non_inf_ind]
        # update wltmp
        wltmp .= g[2n_non_inf_ind+1:3n_non_inf_ind]
    end
    llh2[iter] /= num_batches
    β[ind_l, :] .= βtmp
    wl[ind_l, :] .= wltmp
end

"""
main function for low quality data, train only
"""
function RVM!(
    model::MnRVModel{T}, XL::AbstractMatrix{T}, t::AbstractMatrix{T},
    α::AbstractMatrix{T}, β::AbstractMatrix{T};
    rtol::Float64=1e-6, atol::Float64=1e-8,
    maxiter::Int64=10000, n_samples::Int64=5000,
    BatchSize::Int64=size(XL, 1)
) where T<:Real
    # Multinomial
    prog, wl, whsamples, llh, num_batches, ind, n_ind, n, d, K = setup(
        model, XL, β, t, BatchSize
    )
    β = β[ind, :]
    XL = XL[:, ind]
    for iter ∈ 2:maxiter
        ind_l = unique!([item[1] for item in findall(β .< (1/rtol))])
        n_ind_l = size(ind_l, 1)
        if n_ind_l == 0
            ProgressMeter.finish!(prog, spinner = '✓')
            return model
        end
        epoch!(
            β, wl, XL, t, whsamples, ind_l,
            llh, rtol, atol, maxiter,
            BatchSize, num_batches
        )
        incr = (llh[iter] - llh[iter-1]) / llh[iter-1]
        ProgressMeter.next!(
            prog;
            showvalues = [(:iter,iter-1), (:incr,incr)]
        )
        if abs(incr) < rtol || iter == maxiter
            iter != maxiter ?
                ProgressMeter.finish!(prog, spinner = '✓') : (
                ProgressMeter.finish!(prog, spinner = '✗');
                @warn "Not converged after $(maxiter) steps.
                    Results may be inaccurate."
            )
            H = zeros(T, n_ind_l, n_ind_l, K, n_samples)
            wl = zeros(T, n_indl_l, K, n_samples)
            for b ∈ 1:num_batches
                if b != num_batches
                    XLtmp = copy(XL[(b-1)*BatchSize+1:b*BatchSize, ind_l])
                    ttmp = copy(t[(b-1)*BatchSize+1:b*BatchSize, :])
                else  # the last batch
                    XLtmp = copy(XL[(b-1)*BatchSize+1:end, ind_l])
                    ttmp = copy(t[(b-1)*BatchSize+1:end, :])
                end
                Threads.@threads for nn ∈ 1:n_samples
                    wltmp = Logit(
                        whsamples[ind_l, nn], wltmp, βtmp,
                        XLtmp, transpose(XLtmp),
                        ttmp, non_inf_ind, atol,
                        maxiter, true
                    )
                    @avx A = exp.(XLtmp * wltmp)
                    A ./= sum(A, dims=2)
                    wl[:, :, nn] .+= wltmp ./ num_batches
                    @inbounds for k ∈ 1:K
                        yk = view(A, :, k)
                        yk .= yk .* (1 .- yk)
                        yk[yk .< 1e-10] .= 0.
                        H[:, :, k, nn] .+= WoodburyInv!(
                            view(βtmp, :, k),
                            Diagonal(sqrt.(yk)) * XLtmp
                        ) ./ num_batches
                    end
                end
            end
            println("done.")
            return MnFusedRVModel(wl, H, ind[ind_l])
        end
    end
end

"""
main function for low quality data, train + predict
"""
function RVM!(
    model::MnRVModel{T}, XL::AbstractMatrix{T}, t::AbstractMatrix{T},
    XLtest::AbstractMatrix{T}, α::AbstractMatrix{T}, β::AbstractMatrix{T};
    rtol::Float64=1e-6, atol::Float64=1e-8,
    maxiter::Int64=100000, n_samples::Int64=5000,
    BatchSize::Int64=size(XL, 1)
) where T<:Real
    # Multinomial
    prog, wl, whsamples, llh, num_batches, ind, n_ind, n, d, K = setup(
        model, XL, β, t, BatchSize
    )
    β = β[ind, :]
    XL = XL[:, ind]
    for iter ∈ 2:maxiter
        ind_l = unique!([item[1] for item in findall(β .< (1/rtol))])
        n_ind_l = size(ind_l, 1)
        if n_ind_l == 0
            ProgressMeter.finish!(prog, spinner = '✓')
            return predict(model, XLtest)
        end
        epoch!(
            β, wl, XL, t, whsamples, ind_l,
            llh, rtol, atol, maxiter,
            BatchSize, num_batches
        )
        incr = (llh[iter] - llh[iter-1]) / llh2[iter-1]
        ProgressMeter.next!(
            prog;
            showvalues = [(:iter,iter-1), (:incr,incr)]
        )
        if abs(incr) < rtol || iter == maxiter
            iter != maxiter ?
                ProgressMeter.finish!(prog, spinner = '✓') : (
                ProgressMeter.finish!(prog, spinner = '✗');
                @warn "Not converged after $(maxiter) steps.
                    Results may be inaccurate."
            )
            XLtest = XLtest[:, ind[ind_l]]
            predictions = zeros(T, size(XLtest, 1), K)
            for b ∈ 1:num_batches
                if b != num_batches
                    XLtmp = copy(XL[(b-1)*BatchSize+1:b*BatchSize, ind_l])
                    ttmp = copy(t[(b-1)*BatchSize+1:b*BatchSize, :])
                else  # the last batch
                    XLtmp = copy(XL[(b-1)*BatchSize+1:end, ind_l])
                    ttmp = copy(t[(b-1)*BatchSize+1:end, :])
                end
                p = eachslice(whsamples, dims=3) |> Map(
                    x -> Logit(x[ind_l, :], wltmp, βtmp,
                    XLtmp, transpose(XLtmp), XLtest,
                    ttmp, non_inf_ind, atol,
                    maxiter, true)
                ) |> Broadcasting() |> Folds.sum
                p ./= n_samples
                predictions .+= p
            end
            predictions ./= num_batches
            println("done.")
            return predictions
        end
    end
end

"""
GD for wl
"""
function grad!(
    wh::AbstractMatrix{T}, wl::AbstractMatrix{T}, X::AbstractMatrix{T},
    Xt::AbstractMatrix{T}, t::AbstractMatrix{T}, ind::AbstractVector{Int64},
    wp::AbstractMatrix{T}, g::AbstractMatrix{T}, gp::AbstractMatrix{T},
    A::AbstractMatrix{T}, logY::AbstractMatrix{T}, Y::AbstractMatrix{T},
    llhp::Float64, η::AbstractVector{T}
) where T <: Real
    mul!(g, Xt, t .- Y)
    g[ind] .-= @views α[ind] .* wl[ind]
    copyto!(wp, wl)
    wl[ind] .+= @views g[ind] .* η
    mul!(A, X, wl .+ wh)
    @avx logY .= A .- log.(sum(exp.(A), dims=2))
    llh = @views -0.5sum(α[ind] .* (wl[ind]).^ 2) + sum(t .* logY)
    while !(llh - llhp > 0)
        η .*= 0.8
        wl .= wp .+ g .* η
        mul!(A, X, wl .+ wh)
        @avx logY .= A .- log.(sum(exp.(A), dims=2))
        llh = @views -0.5sum(α[ind] .* (wl[ind]).^ 2) + sum(t .* logY) #+ 0.5sum(log.(α))
    end
    @avx Y .= exp.(logY)
    return llh
end

"""
finding mode of wl at each wh
"""
function Logit(
    wh::AbstractMatrix{T}, wl::AbstractMatrix{T},
    α::AbstractMatrix{T}, X::AbstractMatrix{T},
    Xt::AbstractMatrix{T}, t::AbstractMatrix{T},
    ind::AbstractArray{Int64},
    tol::Float64, maxiter::Int64, is_final::Bool=false
) where T<:Real
    n, d = size(X)
    K = size(t, 2)
    wp, g, gp = (zeros(T, d, K) for _ = 1:3)
    A, Y, logY = (Matrix{T}(undef, n, K) for _ = 1:3)
    mul!(A, X, wl .+ wh)
    @avx logY .= A .- log.(sum(exp.(A), dims=2))
    @avx Y .= exp.(logY)
    llhp = -Inf
    η = [0.00001]
    for iter = 2:maxiter
        llh = grad!(
            wh, wl, X, Xt, t,
            ind, wp, g, gp, A,
            logY, Y, llhp, η
        )
        if llh - llhp < tol || iter == maxiter
            if iter == maxiter
                @warn "Not converged."
            end
            if is_final
                return wl .+ wh
            else
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
                return vcat(wl[ind].^2, g[ind], wl[ind], llh)
            end
        else
            llhp = llh
            η .= @views abs(
                sum((wl[ind] .- wp[ind]) .* (g[ind] .- gp[ind]))
            ) / sum((g[ind] .- gp[ind]) .^ 2)
            copyto!(gp, g)
        end
    end
end

"""
make predictions at each wh
"""
function Logit(
    wh::AbstractMatrix{T}, wl::AbstractMatrix{T},
    α::AbstractMatrix{T}, X::AbstractMatrix{T},
    Xt::AbstractMatrix{T}, t::AbstractMatrix{T},
    Xtest::AbstractArray{T}, ind::AbstractArray{Int64},
    tol::Float64, maxiter::Int64
) where T<:Real
    n, d = size(X)
    K = size(t, 2) # number of classes
    wp, g, gp = (zeros(T, d, K) for _ = 1:3)
    A, Y, logY = (Matrix{T}(undef, n, K) for _ = 1:3)
    mul!(A, X, wl .+ wh)
    @avx logY .= A .- log.(sum(exp.(A), dims=2))
    @avx Y .= exp.(logY)
    llhp = -Inf
    η = [0.00001]
    for iter = 2:maxiter
        llh = grad!(
            wh, wl, X, Xt, t,
            ind, wp, g, gp, A,
            logY, Y, llhp, η
        )
        if llh - llhp < tol || iter == maxiter
            if iter == maxiter
                @warn "Not converged."
            end
            H = zeros(T, d, d, K)
            @inbounds for k ∈ 1:K
                yk = view(Y, :, k)
                yk .= yk .* (1 .- yk)
                yk[yk .< 1e-10] .= 0.
                αk = view(α, :, k)
                H[:, :, k] .= predict(Xtest, wk,
                    WoodburyInv!(
                        αk,
                        Diagonal(sqrt.(yk)) * X
                    )
                )
            end
            return predict(Xtest, wl .+ wh, H)
        else
            llhp = llh
            η .= @views abs(
                sum((wl[ind] .- wp[ind]) .* (g[ind] .- gp[ind]))
            ) / sum((g[ind] .- gp[ind]) .^ 2)
            copyto!(gp, g)
        end
    end
end
