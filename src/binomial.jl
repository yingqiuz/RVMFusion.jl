# interface
function predict(
    X::AbstractMatrix{T}, w::AbstractVector{T},
    H::AbstractMatrix{T}, ind::AbstractVector{Int64}
) where T <: Real
    Xview = @view X[:, ind]
    p = (1 .+ π .* diag(Xview * H * transpose(Xview)) ./ 8).^(-0.5) .* (Xview * w)
    p .= LoopVectorization.@avx 1 ./ (1 .+ exp.(-1 .* p))
end

function predict!(
    a::AbstractVector{T}, X::AbstractMatrix{T}, w::AbstractVector{T},
    Xt::AbstractMatrix{T}, H::AbstractMatrix{T}#, ind::AbstractArray{Int64}
) where T <: Real
    #Xview = @view X[:, ind]
    a .= (1 .+ π .* diag(X * H * Xt) ./ 8).^(-0.5) .* (X * w)
    a .= LoopVectorization.@avx 1 ./ (1 .+ exp.(-1 .* a))
    return a
end

# core algorithm
function RVM!(
    X::AbstractMatrix{T}, t::AbstractVector{T}, α::AbstractVector{T};
    rtol::Float64=1e-6, atol=1e-8, maxiter::Int64=10000
) where T<:Real
    n = size(X, 1)
    d = size(X, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))

    # preallocate type-II likelihood (evidence) vector
    llh2 = Vector{T}(undef, maxiter)
    fill!(llh2, -Inf)
    # Hessian
    # pre-allocate memories
    a, y = (Vector{T}(undef, n) for _ = 1:2)
    w = randn(T, d)
    h = ones(T, n)
    h[findall(iszero, t)] .= -1.0
    ind_nonzero = findall(x -> x > 1e-3, std(X, dims=1)[:])
    prog = ProgressUnknown(
        "training on high quality data...",
        spinner=true
    )
    for iter = 2:maxiter
        ind_h = findall(α .< 10000) # index of nonzeros
        ind = ind_h[findall(in(ind_nonzero), ind_h)]
        αtmp = @view α[ind]
        wtmp = @view w[ind]
        Xtmp = @view X[:, ind]
        n_ind = size(ind, 1)
        #H = Matrix{T}(undef, n_ind, n_ind)
        # find posterior w - mode and hessian
        llh2[iter] = Logit!(
            wtmp, αtmp, Xtmp, transpose(Xtmp),
            t, atol, maxiter, a, h, y
        )
        llh2[iter] -= 0.5 * n_ind * log(2π)
        #llh2[iter] += 0.5logdet(H)
        incr = abs(llh2[iter] - llh2[iter-1]) / abs(llh2[iter-1])
        ProgressMeter.next!(
            prog;
            showvalues = [(:iter,iter-1), (:incr,incr)]
        )
        if incr < rtol
            ProgressMeter.finish!(prog, spinner = '✓')
            H = WoodburyInv!(αtmp, Diagonal(sqrt.(y .* (1 .- y))) * Xtmp)
            return w[ind], convert(Array{T}, Symmetric(H)), ind
        end
        #Σ = Hermitian(H) \ I
        #αtmp .= 1 ./ (wtmp.^2 .+ diag(H)) #(1 .- αtmp .* diag(H)) ./ (wtmp.^2)
    end
    ProgressMeter.finish!(prog, spinner = '✗')
    @warn "Not converged after $(maxiter) iterations."
end

"""
train and predict
"""
function RVM!(
    XH::AbstractMatrix{T}, XL::AbstractMatrix{T},
    t::AbstractVector{T}, XLtest::AbstractMatrix{T},
    α::AbstractVector{T}, β::AbstractVector{T};
    rtol::Float64=1e-6, atol::Float64=1e-8,
    maxiter::Int64=10000, n_samples::Int64=2000
) where T<:Real
    n, d = size(XL)
    # should add more validity checks
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))
    wh, H, ind = RVM!(XH, t, α; rtol=rtol, atol=atol, maxiter=maxiter)
    # preallocate type-II likelihood (evidence) vector
    evi = Vector{T}(undef, maxiter)
    fill!(evi, -Inf)
    #@show fit(Histogram, std(XL, dims=1)[:])
    ind_nonzero = findall(in(findall(x -> x > 1e-3, std(XL, dims=1)[:])), ind)
    ind_h = ind[ind_nonzero]
    #@show ind_h ind
    #@show ind_nonzero
    # now for the lower quality # need a sampler
    # allocate memory
    h = ones(T, n)
    h[findall(iszero, t)] .= -1.0
    println("Generate posterior samples of wh...")
    whsamples = rand(
        MvNormal(
            wh,
            H
        ),
        n_samples
    )
    whsamples = whsamples[ind_nonzero, :]
    XLtmp = view(XL, :, ind_h)
    XLtesttmp = view(XLtest, :, ind_h)
    βtmp = view(β, ind_h)
    #whsamples[ind_h, :] .= rand(MvNormal(wh, H), n_samples)
    prog = ProgressUnknown(
        "training on low quality data...",
        spinner=true
    )
    for iter = 2:maxiter
        # the posterior or MLE solution of wl
        ind_l = findall(βtmp .< 10000) # optional
        n_ind = size(ind_l, 1)
        β2 = copy(βtmp[ind_l])
        XL2 = copy(XLtmp[:, ind_l])
        XL2t = transpose(XL2)
        #subind = 1 + (iter % 5) * 1000 : 1000 + (iter % 5) * 1000
        @views g = whsamples[ind_l, :] |> eachcol |>
        Map(
            x -> Logit(
                x, β2, XL2, XL2t, t,
                h, atol, maxiter
            )
        ) |> Broadcasting() |> Folds.sum
        g ./= n_samples
        evi[iter] = g[end] + 0.5sum(log.(β2)) - 0.5n_ind * log(2π)
        @views βtmp[ind_l] .= (1 .- β2 .* g[n_ind+1:2n_ind]) ./ g[1:n_ind]
        #incr = maximum(abs.(βtmp .- βp) ./ abs.(βp))
        incr = abs(evi[iter] - evi[iter-1]) / abs(evi[iter-1])
        ProgressMeter.next!(
            prog;
            showvalues = [(:iter,iter-1), (:incr,incr)]
        )
        #@info "llh2" evi[iter]
        if incr < rtol
            ProgressMeter.finish!(prog, spinner = '✓')
            XLtest2 = copy(XLtesttmp[:, ind_l])
            XLtest2t = transpose(XLtest2)
            @views y = whsamples[ind_l, :] |> eachcol |>
            Map(
                x -> Logit(
                    x, β2, XL2, XL2t, t,
                    XLtest2, XLtest2t, h, atol, maxiter
                )
            ) |> Broadcasting() |> Folds.sum
            y ./= n_samples
            return y
        end
        #copyto!(βp, βtmp)
    end
    ProgressMeter.finish!(prog, spinner = '✗')
    @warn "Not converged after $(maxiter) iterations."
    return [NaN]
end

function Logit!(
    w::AbstractVector{T}, α::AbstractVector{T},
    X::AbstractMatrix{T}, Xt::AbstractMatrix{T},
    t::AbstractVector{T}, tol::Float64,
    maxiter::Int64, a::AbstractVector{T}, h::AbstractVector{T},
    y::AbstractVector{T}
) where T<:Real

    n = size(X, 1)
    d = size(X, 2)
    g, gp = (zeros(T, d) for _ ∈ 1:2)
    mul!(a, X, w)
    llhp = -Inf; llh = -Inf
    wp = similar(w)
    y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
    r  = [0.00001]
    for iter = 2:maxiter
        # update Hessian
        #H .= Xt * Diagonal(y .* (1 .- y)) * X
        #add_diagonal!(H, α)
        # update gradient
        mul!(g, Xt, t .- y)
        g .-= α .* w
        #@info "g" g
        #ldiv!(qr(H), g)
        # update w
        copyto!(wp, w)
        w .+= g .* r
        mul!(a, X, w)
        LoopVectorization.@avx llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* w .^ 2)
        while !(llh - llhp > 0.)
            r ./= 2
            w .= wp .+ g .* r
            mul!(a, X, w)
            LoopVectorization.@avx llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* w .^ 2)
        end
        LoopVectorization.@avx y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
        if llh - llhp < tol
            llh += 0.5sum(log.(α))
            WoodburyInv!(g, α, Diagonal(sqrt.(y .* (1 .- y))) * X)
            #H .= Xt * Diagonal(y .* (1 .- y)) * X
            #add_diagonal!(H, α)
            α .= (1 .- α .* g) ./ (w.^2)
            return llh
        end
        r .= sum((w .- wp) .* (g .- gp))
        r .= abs.(r) ./ sum((g .- gp) .^ 2)
        llhp = llh
        copyto!(gp, g)
    end
    @warn "Not converged in finding the posterior of wh."
    return NaN
end

function Logit(
    wh::AbstractVector{T}, α::AbstractVector{T},
    X::AbstractMatrix{T}, Xt::AbstractMatrix{T},
    t::AbstractVector{T}, #Xtest::AbstractArray{T},
    h::AbstractVector{T}, tol::Float64, maxiter::Int64
) where T<:Real
    # need a sampler
    n, d = size(X)
    wp, wl, g, gp = (zeros(T, d) for _ = 1:4)
    #H = Matrix{T}(undef, d, d)
    a, y = (Vector{T}(undef, n) for _ = 1:2)
    #wl = copy(wh)
    mul!(a, X, wh)
    llhp = -Inf; llh = -Inf
    LoopVectorization.@avx y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
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
        mul!(a, X, wl .+ wh)
        LoopVectorization.@avx llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* wl .^ 2)
        while !(llh - llhp > 0.)
            r ./= 2
            wl .= wp .+ g .* r
            mul!(a, X, wl .+ wh)
            LoopVectorization.@avx llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* wl .^ 2)
        end
        LoopVectorization.@avx y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
        if llh - llhp < tol
            WoodburyInv!(g, α, Diagonal(sqrt.(y .* (1 .- y))) * X)
            #predict!(y, Xtest, wl, H, 1:d)
            return vcat(wl.^2, g, llh)#, llh+0.5logdet(H))
        else
            llhp = llh
            r .= abs(sum((wl .- wp) .* (g .- gp))) ./ sum((g .- gp) .^ 2)
        end
    end
    @warn "Not converged in finding the posterior of wh."
    return zeros(T, 2d+1)
end

function Logit(
    wh::AbstractVector{T}, α::AbstractVector{T},
    X::AbstractMatrix{T}, Xt::AbstractMatrix{T},
    t::AbstractVector{T}, Xtest::AbstractMatrix{T}, Xtestt::AbstractMatrix{T},
    h::AbstractVector{T}, tol::Float64, maxiter::Int64
) where T<:Real
    # need a sampler
    n, d = size(X)
    wp, g, gp, wl = (zeros(T, d) for _ = 1:4)
    #H = Matrix{T}(undef, d, d)
    a, y = (Vector{T}(undef, n) for _ = 1:2)
    #wl = copy(wh)
    mul!(a, X, wh)
    llhp = -Inf; llh = -Inf
    LoopVectorization.@avx y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
    r = [0.00001]
    pred = zeros(T, size(Xtest, 1))
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
        LoopVectorization.@avx llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* wl .^ 2)
        while !(llh - llhp > 0.)
            r ./= 2
            wl .= wp .+ g .* r
            mul!(a, X, wl .+ wh)
            LoopVectorization.@avx llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* wl .^ 2)
        end
        LoopVectorization.@avx y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
        if llh - llhp < tol
            H = WoodburyInv!(α, Diagonal(sqrt.(y .* (1 .- y))) * X)
            predict!(pred, Xtest, wl .+ wh, Xtestt, H)
            return pred #, llh+0.5logdet(H))
        else
            llhp = llh
            r .= abs(sum((wl .- wp) .* (g .- gp))) ./ sum((g .- gp) .^ 2)
        end
    end
    @warn "Not converged in finding the posterior of wh."
    return pred
end
