# interface
function RVM(X::Matrix{T}, t::Vector{Int64}, α_init::Float64=1.0;
             kw...) where T<:Real
    n = size(X, 1)
    d = size(X, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))

    K = size(unique(t), 1)
    if K > 2
        α = ones(T, d, K) .* α_init
    elseif K == 2
        α = ones(T, d) .* α_init
    else
        throw(TypeError("Number of classes less than 2."))
    end
    RVM!(X, convert(Vector{T}, t), α; kw...)
end

function predict(
    X::AbstractArray{T}, w::AbstractArray{T},
    H::AbstractArray{T}, ind::AbstractArray{Int64}
) where T <: Real
    Xview = @view X[:, ind]
    p = diag(Xview * H * transpose(Xview))
    p .= (1 .+ π .* p ./ 8).^(-0.5) .* (Xview * w)
    p .= 1 ./ (1 .+ exp.(-1 .* p))
end

function predict!(
    a::AbstractArray{T}, X::AbstractArray{T}, w::AbstractArray{T},
    H::AbstractArray{T}, ind::AbstractArray{Int64}
) where T <: Real
    Xview = @view X[:, ind]
    a .= diag(Xview * H * transpose(Xview))
    a .= (1 .+ π .* a ./ 8).^(-0.5) .* (Xview * w)
    #n = size(a, 1)
    #d = size(H, 1)
    #@tturbo for k = 1:n
    #    Xview = @view X[k, ind]
    #    @inbounds a[k] = Xview * H * transpose(Xview)
    #end
    #a .*= π/8; a .+= 1.0; a .^= -0.5; a .*= view(X, :, ind) * w
    a .= 1 ./ (1 .+ exp.(-1 .* a))
    return a
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
    elseif K == 2
        α = ones(T, d) .* α_init
        β = ones(T, d) .* β_init
    else
        throw(TypeError("Number of classes less than 2."))
    end
    RVM!(XH, XL, convert(Vector{T}, t), XLtest, α, β; kw...)
end

# core algorithm
function RVM!(X::Matrix{T}, t::Vector{T}, α::Vector{T};
              tol::Float64=1e-5, maxiter::Int64=10000) where T<:Real
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
        llh2[iter], H = Logit!(
            wtmp, αtmp, Xtmp, transpose(Xtmp),
            t, tol, maxiter, a, h, y
        )
        llh2[iter] += 0.5sum(log.(αtmp))
        llh2[iter] += 0.5logdet(H)
        incr = abs(llh2[iter] - llh2[iter-1]) / abs(llh2[iter-1])
        ProgressMeter.next!(
            prog;
            showvalues = [(:iter,iter-1), (:incr,incr)]
        )
        if incr < tol
            ProgressMeter.finish!(prog, spinner = '✓')
            return w[ind], convert(Array{T}, Symmetric(H)), ind
        end
        #Σ = Hermitian(H) \ I
        αtmp .= (1 .- αtmp .* diag(H)) ./ (wtmp.^2)
        #αtmp .= 1 ./ (wtmp.^2 .+ diag(H)) #(1 .- αtmp .* diag(H)) ./ (wtmp.^2)
    end
    ProgressMeter.finish!(prog, spinner = '✗')
    warn("Not converged after $(maxiter) iterations.")
end

"""train only - high + low - useless for now
"""
function RVM!(XH::Matrix{T}, XL::Matrix{T}, t::Vector{T},
    α::Vector{T}, β::Vector{T}; tol::Float64=1e-5,
    maxiter::Int64=10000, n_samples::Int64=1000
) where T<:Real

    n, d = size(XL)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))

    wh, H, ind_h = RVM!(XH, t, α; tol=tol, maxiter=maxiter)
    H = convert(Array{T}, Symmetric(H))
    # preallocate type-II likelihood (evidence) vector
    evi = Vector{T}(undef, maxiter)
    fill!(evi, -Inf)
    # now for the lower quality # need a sampler
    # allocate memory
    h = ones(T, n)
    h[findall(iszero, t)] .= -1.0
    println("Generate posterior samples of wh...")
    whsamples = zeros(T, d, n_samples)
    whsamples[ind_h, :] .= rand(MvNormal(wh, H), n_samples)
    for iter = 2:maxiter
        # the posterior or MLE solution of wl
        ind = findall(1 ./ β .> tol) # optional
        n_ind = size(ind, 1)
        βtmp = @view β[ind]
        XLtmp = @view XL[:, ind]
        XLtmpt = transpose(XLtmp)
        #results = Vector{Float64}(undef, n_ind*3 + 1)
        #Threads.@threads for i = 1:n_samples
            #@inbounds results[:, i] .+= Logit(view(whsamples, ind, i),
                #βtmp, XLtmp, XLtmpt, t, h, tol, maxiter)
        #end
        #@tturbo for i = 1:n_samples
        #    g += Logit(whsamples[ind, i], βtmp, XL[:, ind],
        #               t, h, tol, maxiter)
        #end
        #g = ThreadsX.sum(Logit(view(whsamples[ind, i]), βtmp,
        #                 XLtmp, XLtmpt, t, h, tol, maxiter)
        #                 for i = 1:n_samples)
        @views g = whsamples[ind, :] |> eachcol |>
        Map(x -> Logit(x, βtmp, XLtmp, XLtmpt, t, h, tol, maxiter)) |>
        Broadcasting() |> Folds.sum
        g ./= n_samples
        @views βtmp .= (1 - βtmp .* g[n_ind+1:2n_ind]) ./ g[1:n_ind]
        #for k = 1:n_ind
        #    @inbounds βtmp[k] = (1 - βtmp[k] * g[k+n_ind]) / g[k]
        #end
        # check convergence
        evi[iter] = g[end] + 0.5sum(log.(βtmp[βtmp .> 0]))
        incr = abs(evi[iter] - evi[iter-1]) / abs(evi[iter-1])
        println("iteration ", iter-1, ", increment is ", incr)
        if incr < tol
            return wh, H, ind_h, β, g[2*n_ind+1:end-1], ind
            break
        end
    end
    warn("Not converged. Results may be inaccurate.")
end

"""
train and predict
"""
function RVM!(
    XH::Matrix{T}, XL::Matrix{T}, t::Vector{T}, XLtest::Matrix{T},
    α::Vector{T}, β::Vector{T}; tol::Float64=1e-5,
    maxiter::Int64=10000, n_samples::Int64=2000
) where T<:Real

    n, d = size(XL)
    # should add more validity checks
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))
    wh, H, ind = RVM!(XH, t, α; tol=tol, maxiter=maxiter)
    # preallocate type-II likelihood (evidence) vector
    evi = Vector{T}(undef, maxiter)
    fill!(evi, -Inf)
    @show fit(Histogram, std(XL, dims=1)[:])
    ind_nonzero = findall(in(findall(x -> x > 1e-3, std(XL, dims=1)[:])), ind)
    ind_h = ind[ind_nonzero]
    @show ind_h ind
    # now for the lower quality # need a sampler
    # allocate memory
    h = ones(T, n)
    h[findall(iszero, t)] .= -1.0
    println("Generate posterior samples of wh...")
    @views whsamples = rand(
        MvNormal(
            wh[ind_nonzero],
            H[ind_nonzero, ind_nonzero]
        ),
        n_samples
    )
    XLtmp = copy(XL[:, ind_h])
    XLtesttmp = copy(XLtest[:, ind_h])
    βtmp = copy(β[ind_h])
    βp = similar(βtmp)
    #whsamples[ind_h, :] .= rand(MvNormal(wh, H), n_samples)
    prog = ProgressUnknown(
        "training on low quality data...",
        spinner=true
    )
    for iter = 2:maxiter
        # the posterior or MLE solution of wl
        ind_l = findall(βtmp .< 10000) # optional
        n_ind = size(ind_l, 1)
        β2 = @view βtmp[ind_l]
        XL2 = @view XLtmp[:, ind_l]
        XL2t = transpose(XL2)
        XLtest2 = @view XLtesttmp[:, ind_l]
        #subind = 1 + (iter % 5) * 1000 : 1000 + (iter % 5) * 1000
        @views g = whsamples[ind_l, :] |> eachcol |>
        Map(
            x -> Logit(
                x, β2, XL2, XL2t, t,
                h, tol, maxiter
            )
        ) |> Broadcasting() |> Folds.sum
        g ./= n_samples
        evi[iter] = g[end] #+ 0.5sum(log.(β2))
        @views β2 .= (1 .- β2 .* g[n_ind+1:2n_ind]) ./ g[1:n_ind]
        #incr = maximum(abs.(βtmp .- βp) ./ abs.(βp))
        incr = abs(evi[iter] - evi[iter-1]) / abs(evi[iter-1])
        ProgressMeter.next!(
            prog;
            showvalues = [(:iter,iter-1), (:incr,incr)]
        )
        if incr < tol
            ProgressMeter.finish!(prog, spinner = '✓')
            @views g = whsamples[ind_l, :] |> eachcol |>
            Map(
                x -> Logit(
                    x, β2, XL2, XL2t, t,
                    XLtest2, h, tol, maxiter
                )
            ) |> Broadcasting() |> Folds.sum
            g ./= n_samples
            return g[2n_ind+1:end]
        end
        copyto!(βp, βtmp)
    end
    ProgressMeter.finish!(prog, spinner = '✗')
    @warn "Not converged after $(maxiter) iterations."
end

function Logit!(
    w::AbstractArray{T}, α::AbstractArray{T},
    X::AbstractArray{T}, Xt::AbstractArray{T},
    t::AbstractArray{T}, tol::Float64,
    maxiter::Int64, a::AbstractArray{T}, h::AbstractArray{T},
    y::AbstractArray{T}
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
        #ldiv!(qr(H), g)
        # update w
        copyto!(wp, w)
        w .+= g .* r
        mul!(a, X, w)
        llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* w .^ 2)
        while llh - llhp < 0.0
            g ./= 2
            w .= wp .+ g .* r
            mul!(a, X, w)
            llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* w .^ 2)
        end
        y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
        if llh - llhp < tol
            #H = WoodburyInv!(α, Diagonal(sqrt.(y .* (1 .- y))) * X)
            #H .= Xt * Diagonal(y .* (1 .- y)) * X
            #add_diagonal!(H, α)
            return llh, WoodburyInv!(α, Diagonal(sqrt.(y .* (1 .- y))) * X)
        end
        r .= sum((w .- wp) .* (g .- gp))
        r .= abs.(r) ./ sum((g .- gp) .^ 2)
        llhp = llh
        copyto!(gp, g)
    end
    @warn "Not converged in finding the posterior of wh."
end

function Logit(
    wh::AbstractArray{T}, α::AbstractArray{T},
    X::AbstractArray{T}, Xt::AbstractArray{T},
    t::AbstractArray{T}, h::AbstractArray{T},
    tol::Float64, maxiter::Int64
) where T<:Real
    # need a sampler
    n, d = size(X)
    wp, g = (zeros(T, d) for _ = 1:2)
    H = Matrix{T}(undef, d, d)
    a, y = (Vector{T}(undef, n) for _ = 1:2)
    wl = copy(wh)
    mul!(a, X, wh)
    llhp = -Inf; llh = -Inf
    for iter = 2:maxiter
        y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
        # update Hessian
        H .= Xt * Diagonal(y .* (1 .- y)) * X
        add_diagonal!(H, α)
        # update gradient
        mul!(g, Xt, t .- y)
        g .-= α .* wl
        ldiv!(qr(H), g)
        # update w
        copyto!(wp, wl)
        wl .+= g
        mul!(a, X, wl)
        llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* wl .^ 2)
        while llh - llhp < 0.0
            g ./= 2
            wl .= wp .+ g
            mul!(a, X, wl)
            llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* wl .^ 2)
        end
        llh - llhp > tol ? llhp = llh : break
    end
    # last update of Hessian
    y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
    # update Hessian
    H .= Xt * Diagonal(y .* (1 .- y)) * X
    add_diagonal!(H, α)
    ldiv!(H, qr(H), I(d))  # need to be sure
    return vcat((wl.-wh).^2, diag(H), wl, llh+0.5logdet(H))
end

function Logit(
    wh::AbstractArray{T}, α::AbstractArray{T},
    X::AbstractArray{T}, Xt::AbstractArray{T},
    t::AbstractVector{T}, #Xtest::AbstractArray{T},
    h::AbstractArray{T}, tol::Float64, maxiter::Int64
) where T<:Real
    # need a sampler
    n, d = size(X)
    wp, g, gp = (zeros(T, d) for _ = 1:3)
    #H = Matrix{T}(undef, d, d)
    a, y = (Vector{T}(undef, n) for _ = 1:2)
    wl = copy(wh)
    mul!(a, X, wh)
    llhp = -Inf; llh = -Inf
    y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
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
        mul!(a, X, wl)
        llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* wl .^ 2)
        while llh - llhp < 0.0
            g ./= 2
            wl .= wp .+ g .* r
            mul!(a, X, wl)
            llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* wl .^ 2)
        end
        y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
        if llh - llhp < tol
            WoodburyInv!(g, α, Diagonal(sqrt.(y .* (1 .- y))) * X)
            #predict!(y, Xtest, wl, H, 1:d)
            return vcat((wl.-wh).^2, g)#, llh+0.5logdet(H))
        else
            llhp = llh
            r .= abs(sum((wl .- wp) .* (g .- gp))) ./ sum((g .- gp) .^ 2)
        end
    end
    @warn "Not converged in finding the posterior of wh."
end

function Logit(
    wh::AbstractArray{T}, α::AbstractArray{T},
    X::AbstractArray{T}, Xt::AbstractArray{T},
    t::AbstractVector{T}, Xtest::AbstractArray{T},
    h::AbstractArray{T}, tol::Float64, maxiter::Int64
) where T<:Real
    # need a sampler
    n, d = size(X)
    wp, g, gp = (zeros(T, d) for _ = 1:3)
    #H = Matrix{T}(undef, d, d)
    a, y = (Vector{T}(undef, n) for _ = 1:2)
    wl = copy(wh)
    mul!(a, X, wh)
    llhp = -Inf; llh = -Inf
    y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
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
        mul!(a, X, wl)
        llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* wl .^ 2)
        while llh - llhp < 0.0
            g ./= 2
            wl .= wp .+ g .* r
            mul!(a, X, wl)
            llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* wl .^ 2)
        end
        y .= 1.0 ./ (1.0 .+ exp.(-1.0 .* a))
        if llh - llhp < tol
            H = WoodburyInv!(α, Diagonal(sqrt.(y .* (1 .- y))) * X)
            predict!(y, Xtest, wl, H, 1:d)
            return vcat((wl.-wh).^2, diag(H), y) #, llh+0.5logdet(H))
        else
            llhp = llh
            r .= abs(sum((wl .- wp) .* (g .- gp))) ./ sum((g .- gp) .^ 2)
        end
    end
    @warn "Not converged in finding the posterior of wh."
end
