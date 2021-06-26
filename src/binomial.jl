# define types
struct BnRVModel{T<:Real} <: Model
    w::AbstractArray{T, 1}
    H::AbstractArray{T, 2}
    ind::AbstractArray{Int64, 1}
end

struct BnFusedRVModel{T<:Real} <: Model
    w::AbstractArray{T, 2}
    H::AbstractArray{T, 3}
    ind::AbstractArray{Int64, 1}
end

# predict function
function predict(
    model::BnRVModel{T}, X::AbstractMatrix{T}
) where T <: Real
    Xview = @view X[:, model.ind]
    p = (1 .+ π .* diag(Xview * model.H * transpose(Xview)) ./ 8).^(-0.5) .* (Xview * model.w)
    p .= @avx 1 ./ (1 .+ exp.(-1 .* p))
    return p
end

function predict(
    model::BnFusedRVModel{T}, X::AbstractMatrix{T}
) where T <: Real
    Xnew = copy(X[:, model.ind])
    n = size(X, 1)
    n_samples = size(model.w, 2)
    p = 1:n_samples |> Map(
        k -> predict(Xnew, model.w[:, k], model.H[:, :, k])
    ) |> Broadcasting() |> Folds.sum
    return p ./ n_samples
end

function predict(
    X::AbstractMatrix{T}, w::AbstractVector{T}, H::AbstractMatrix{T}
) where T <: Real
    n, d = size(X)
    p = Vector{T}(undef, n)
    fill!(p, 1.)
    @turbo for nn ∈ 1:n, i ∈ 1:d, j ∈ 1:d
        p[nn] += (π / 8) * X[nn, i] * H[i, j] * X[nn, j]
    end
    p .= p.^(-0.5)
    p .*= X * w
    return @avx 1 ./ (1 .+ exp.(-1 .* p))
end

# core algorithm
function RVM!(
    X::AbstractMatrix{T}, t::AbstractVector{T}, α::AbstractVector{T};
    rtol::Float64=1e-6, atol=1e-6, maxiter::Int64=10000,
    BatchSize::Int64=size(X, 1), StepSize::Float64=0.0001
) where T<:Real
# default full batch
    n = size(X, 1)
    d = size(X, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))
    h = ones(T, n)
    h[findall(iszero, t)] .= -1.0
    # preallocate type-II likelihood (evidence) vector
    llh2 = zeros(T, maxiter)
    llh2[1] = -Inf
    w, αsum = (zeros(T, d) for _ = 1:2)# .+ 1e-8
    # pre-allocate memories
    num_batches = n ÷ BatchSize
    a1, y1 = (Vector{T}(undef, BatchSize) for _ = 1:2)
    a2, y2 = (Vector{T}(undef, n - BatchSize * (num_batches-1)) for _ = 1:2)
    ind_nonzero = findall(x -> x > 1e-3, std(X, dims=1)[:])
    println("Setup done.")
    for iter ∈ 2:maxiter
        ind_h = findall(α .< (1/rtol)) # index of nonzeros
        ind = ind_h[findall(in(ind_nonzero), ind_h)]
        αtmp = copy(α[ind])
        wtmp = copy(w[ind])
        g, gp, wp = (similar(αtmp) for _ = 1:3)
        n_ind = size(ind, 1)
        # loop through batches
        @showprogress 0.5 "epoch $(iter) " for b ∈ 1:num_batches
            #copyto!(αgradp, αgrad)
            if b != num_batches
                Xtmp = copy(X[(b-1)*BatchSize+1:b*BatchSize, ind])
                ttmp = copy(t[(b-1)*BatchSize+1:b*BatchSize])
                llh2[iter] += Logit!(
                    wtmp, αtmp, Xtmp, transpose(Xtmp),
                    ttmp, atol, maxiter, a1, y1, g, gp, wp
                )
            else
                Xtmp = copy(X[(b-1)*BatchSize+1:end, ind])
                ttmp = copy(t[(b-1)*BatchSize+1:end])
                llh2[iter] += Logit!(
                    wtmp, αtmp, Xtmp, transpose(Xtmp),
                    ttmp, atol, maxiter, a2, y2, g, gp, wp
                )
            end
        end
        α[ind] .= αtmp
        w[ind] .= wtmp
        # finish all mini batches
        llh2[iter] /= num_batches
        # last iteration
        incr = (llh2[iter] - llh2[iter-1]) / llh2[iter-1]
        println("epoch ", iter, " done. incr ", incr)
        if abs(incr) < rtol || iter == maxiter
            if iter == maxiter
                @warn "Not converged after $(maxiter) iterations."
            end
            H = zeros(T, n_ind, n_ind)
            y = view(X, :, ind) * wtmp
            @avx y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* y))
            for b ∈ 1:num_batches
                Xview = view(X, (b-1)*BatchSize+1:b*BatchSize, ind)
                yview = view(y, (b-1)*BatchSize+1:b*BatchSize)
                H .+= WoodburyInv!(
                    αtmp,
                    Diagonal(sqrt.(yview .* (1 .- yview))) * Xview
                )
            end
            H ./= num_batches
            println("done.")
            return BnRVModel(wtmp, convert(Array{T}, Symmetric(H)), ind)
        end
    end
end

"""
train only
"""
function RVM!(
    XH::AbstractMatrix{T}, XL::AbstractMatrix{T},
    t::AbstractVector{T}, #XLtest::AbstractMatrix{T},
    α::AbstractVector{T}, β::AbstractVector{T};
    rtol::Float64=1e-6, atol::Float64=1e-6,
    maxiter::Int64=10000, n_samples::Int64=5000,
    BatchSize::Int64=size(XL, 1)#, StepSize::Float64=0.01
) where T<:Real
    n, d = size(XL)
    # should add more validity checks
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))
    num_batches = n ÷ BatchSize
    # cleaner data
    model = RVM!(
        XH, t, α;
        rtol=rtol, atol=atol,
        maxiter=maxiter, BatchSize=BatchSize
    )
    # preallocate type-II likelihood (evidence) vector
    llh = zeros(T, maxiter)
    llh[1] = -Inf
    # remove zero columns
    ind_nonzero = findall(
        in(findall(x -> x > 1e-3, std(XL, dims=1)[:])),
        model.ind
    )
    ind_h = model.ind[ind_nonzero]
    # prune wh and H
    wh = model.w[ind_nonzero]
    H = model.H[ind_nonzero, ind_nonzero]
    #@show ind_h ind
    #@show ind_nonzero
    println("Generate posterior samples of wh...")
    whsamples = rand(
        MvNormal(
            wh,
            Symmetric(H)
        ),
        n_samples
    )
    XL = XL[:, ind_h]
    #XLtest = XLtest[:, ind_h]
    β = β[ind_h]
    println("Setup done.")
    for iter ∈ 2:maxiter
        # the posterior or MLE solution of wl
        ind_l = findall(β .< (1/rtol)) # optional
        n_ind = size(ind_l, 1)
        βtmp = copy(β[ind_l])
        whtmp = copy(whsamples[ind_l, :])
        # iterate through batches
        @showprogress 0.5 "epoch $(iter-1) " for b ∈ 1:num_batches
            if b != num_batches
                XLtmp = copy(XL[(b-1)*BatchSize+1:b*BatchSize, ind_l])
                ttmp = copy(t[(b-1)*BatchSize+1:b*BatchSize])
            else
                XLtmp = copy(XL[(b-1)*BatchSize+1:end, ind_l])
                ttmp = copy(t[(b-1)*BatchSize+1:end])
            end
            g = whtmp |> eachcol |>
            Map(
                x -> Logit(
                    x, βtmp, XLtmp, transpose(XLtmp), ttmp,
                    atol, maxiter#, βgrad
                )
            ) |> Broadcasting() |> Folds.sum
            g ./= n_samples
            llh[iter] += g[end] + 0.5sum(log.(βtmp)) - 0.5n_ind*log(2π)
            βtmp .= @views (1 .- βtmp .* g[n_ind+1:2n_ind]) ./ (g[1:n_ind] + 1e-10)
            #βsum[ind_l] .+= @views g[1:end-1] .^ 2
        end
        β[ind_l] .= βtmp
        llh[iter] /= num_batches
        incr = (llh[iter] - llh[iter-1]) / llh[iter-1]
        if abs(incr) < rtol || iter == maxiter
            if iter == maxiter
                @warn "Not converged after $(maxiter) iterations.
                    Results might be inaccurate."
            end
            wl = zeros(T, n_ind, n_samples)
            H = zeros(T, n_ind, n_ind, n_samples)
            Threads.@threads for col ∈ 1:n_samples
                @inbounds for b ∈ 1:num_batches
                    if b != num_batches
                        XLtmp = copy(XL[(b-1)*BatchSize+1:b*BatchSize, ind_l])
                        ttmp = copy(t[(b-1)*BatchSize+1:b*BatchSize])
                    else
                        XLtmp = copy(XL[(b-1)*BatchSize+1:end, ind_l])
                        ttmp = copy(t[(b-1)*BatchSize+1:end])
                    end
                    wltmp = Logit(
                        whtmp[:, col], βtmp, XLtmp,
                        transpose(XLtmp), ttmp, atol,
                        maxiter, true
                    )
                    wl[:, col] .+= wltmp ./ num_batches
                    #mul!(y, XLtmp, view(wl, :, col))
                    #@avx y .= 1 ./ (1 .+ exp.(-y))
                    H[:, :, col] .+= WoodburyInv!(
                        βtmp,
                        Diagonal(
                            sqrt.(
                                @avx 1 ./ (1 .+ exp.(XLtmp * wltmp))
                            )
                        ) * XLtmp
                    ) ./ num_batches
                end
                #wl[:, col] .+=  view(whtmp, :, col)
                #H[:, :, col] ./= num_batches
            end
            return BnFusedRVModel(wl, H, ind_h[ind_l])
        end
    end
end

function Logit!(
    w::AbstractVector{T}, α::AbstractVector{T},
    X::AbstractMatrix{T}, Xt::AbstractMatrix{T},
    t::AbstractVector{T}, tol::Float64,
    maxiter::Int64, a::AbstractVector{T},
    y::AbstractVector{T}, g::AbstractVector{T},
    gp::AbstractVector{T}, wp::AbstractVector{T}
) where T<:Real
    n = size(X, 1)
    d = size(X, 2)
    #gp = zeros(T, d)
    mul!(a, X, w)
    llhp = -Inf; llh = -Inf
    #wp = similar(w)
    @avx y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
    r  = [0.0001]
    for iter = 2:maxiter
        mul!(g, Xt, t .- y)
        g .-= α .* w
        #@info "g" g
        #ldiv!(qr(H), g)
        # update w
        copyto!(wp, w)
        w .+= g .* r
        mul!(a, X, w)
        @avx llh = -sum(log1p.(exp.((1 .- 2 .* t) .* a))) - 0.5sum(α .* w .^ 2)
        while !(llh - llhp > 0.)
            r *= 0.8
            w .= wp .+ g .* r
            mul!(a, X, w)
            @avx llh = -sum(log1p.(exp.((1 .- 2 .* t) .* a))) - 0.5sum(α .* w .^ 2)
        end
        @avx y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
        if llh - llhp < tol || iter == maxiter
            llh += 0.5sum(log.(α)) - 0.5d*log(2π)
            WoodburyInv!(g, α, Diagonal(sqrt.(y .* (1 .- y))) * X)
            α .= (1 .- α .* g) ./ (w .^ 2 .+ 1e-10)
            #g .= 0.5 .* (w.^2 .+ g .- 1 ./ α)
            if iter == maxiter
                @warn "Not converged in finding the posterior of wh."
            end
            return llh
        end
        r .= sum((w .- wp) .* (g .- gp))
        r .= abs.(r) ./ sum((g .- gp) .^ 2)
        llhp = llh
        copyto!(gp, g)
    end
end

function Logit(
    wh::AbstractVector{T}, α::AbstractVector{T},
    X::AbstractMatrix{T}, Xt::AbstractMatrix{T},
    t::AbstractVector{T}, tol::Float64, maxiter::Int64,
    is_final::Bool=false
) where T<:Real
    # need a sampler
    n, d = size(X)
    wp, wl, g, gp = (zeros(T, d) for _ = 1:4)
    #H = Matrix{T}(undef, d, d)
    a, y = (Vector{T}(undef, n) for _ = 1:2)
    #wl = copy(wh)
    mul!(a, X, wh .+ wl)
    llhp, llh = (-Inf, -Inf)
    @avx y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
    r = [0.0001]
    for iter = 2:maxiter
        # update gradient
        copyto!(gp, g)
        mul!(g, Xt, t .- y)
        g .-= α .* wl
        #ldiv!(factorize(H), g)
        # update w
        copyto!(wp, wl)
        wl .+= g .* r
        mul!(a, X, wl .+ wh)
        @avx llh = -sum(log1p.(exp.((1.0 .- 2.0 .* t) .* a))) - 0.5sum(α .* wl .^ 2)
        while !(llh - llhp > 0.)
            r ./= 2
            wl .= wp .+ g .* r
            mul!(a, X, wl .+ wh)
            @avx llh = -sum(log1p.(exp.((1.0 .- 2.0 * t) .* a))) - 0.5sum(α .* wl .^ 2)
        end
        @avx y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
        if llh - llhp < tol || iter == maxiter
            WoodburyInv!(g, α, Diagonal(sqrt.(y .* (1 .- y))) * X)
            if iter == maxiter
                @warn "Not converged in finding the posterior of wl."
            end
            if is_final
                return wl .+ wh
            else
                return vcat(wl.^2, g, llh)
            end
        else
            llhp = llh
            r .= abs(sum((wl .- wp) .* (g .- gp))) ./ sum((g .- gp) .^ 2)
        end
    end
end

# function Logit(
#     wh::AbstractVector{T}, α::AbstractVector{T},
#     X::AbstractMatrix{T}, Xt::AbstractMatrix{T},
#     t::AbstractVector{T}, Xtest::AbstractMatrix{T}, Xtestt::AbstractMatrix{T},
#     tol::Float64, maxiter::Int64
# ) where T<:Real
#     # need a sampler
#     n, d = size(X)
#     wp, g, gp, wl = (zeros(T, d) for _ = 1:4)
#     #H = Matrix{T}(undef, d, d)
#     a, y = (Vector{T}(undef, n) for _ = 1:2)
#     #wl = copy(wh)
#     mul!(a, X, wh)
#     llhp = -Inf; llh = -Inf
#     LoopVectorization.@avx y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
#     r = [0.00001]
#     pred = zeros(T, size(Xtest, 1))
#     for iter = 2:maxiter
#         # update gradient
#         copyto!(gp, g)
#         mul!(g, Xt, t .- y)
#         g .-= α .* wl
#         #ldiv!(factorize(H), g)
#         # update w
#         copyto!(wp, wl)
#         wl .+= g .* r
#         mul!(a, X, wl .+ wh)
#         LoopVectorization.@avx llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* wl .^ 2)
#         while !(llh - llhp > 0.)
#             r ./= 2
#             wl .= wp .+ g .* r
#             mul!(a, X, wl .+ wh)
#             LoopVectorization.@avx llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* wl .^ 2)
#         end
#         LoopVectorization.@avx y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
#         if llh - llhp < tol || iter == maxiter
#             if iter == maxiter
#                 @warn "Not converged in finding the posterior of wl."
#             end
#             H = WoodburyInv!(α, Diagonal(sqrt.(y .* (1 .- y))) * X)
#             #predict!(pred, Xtest, wl .+ wh, Xtestt, H)
#             return wl#, H #, llh+0.5logdet(H))
#         else
#             llhp = llh
#             r .= abs(sum((wl .- wp) .* (g .- gp))) ./ sum((g .- gp) .^ 2)
#         end
#     end
# end
