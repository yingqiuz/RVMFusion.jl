# define types
struct BnRVModel{T<:Real} <: Model
    w::AbstractArray{T, 1}
    H::AbstractArray{T, 2}
    ind::AbstractArray{Int, 1}
end

struct BnFusedRVModel{T<:Real} <: Model
    w::AbstractArray{T, 2}
    H::AbstractArray{T, 3}
    ind::AbstractArray{Int, 1}
end

# predict function
function predict(
    model::BnRVModel{T}, X::AbstractMatrix{T}
) where T <: Real
    Xview = @view X[:, model.ind]
    p = (1 .+ diag(Xview * model.H * transpose(Xview)) .* π ./ 8).^(-0.5f0) .*
        (Xview * model.w)
    return logistic.(p)
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
    X::AbstractMatrix{T}, w::AbstractVector{T}, H::AbstractMatrix{T},
    Xt::AbstractMatrix{T}=transpose(X)
) where T <: Real
    p = (1 .+ diag(X * H * Xt) .* π ./ 8).^(-0.5f0) .*
        (X * w)
    return logistic.(p)
end

function predict(
    X::AbstractMatrix{T}, w::AbstractVector{T}
) where T <: Real
    return logistic.(X * w)
end

# core algorithm
function RVM!(
    X::AbstractMatrix{T}, t::AbstractVector{T}, α::AbstractVector{T};
    rtol::T=convert(T, 1e-5), atol::T=convert(T, 1e-5), maxiter::Int=10000,
    BatchSize::Int=size(X, 1), ϵ::T=convert(T, 1e-8)
) where T<:Real
# default full batch
    n = size(X, 1)
    d = size(X, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))
    #h = ones(T, n)
    #h[findall(iszero, t)] .= -1.0
    # preallocate type-II likelihood (evidence) vector
    llh2 = zeros(T, maxiter)
    llh2[1] = -Inf32
    w = zeros(T, d)
    # pre-allocate memories
    num_batches = convert(Int, round(n / BatchSize))
    a1, y1 = (Vector{T}(undef, BatchSize) for _ = 1:2)
    a2, y2 = (Vector{T}(undef, n - BatchSize * (num_batches-1)) for _ = 1:2)
    ind_nonzero = findall(x -> x > 0.001f0, std(X, dims=1)[:])
    println("Setup done.")
    for iter ∈ 2:maxiter
        ind_h = findall(α .< (1/rtol)) # index of nonzeros
        ind = ind_h[findall(in(ind_nonzero), ind_h)]
        αtmp = copy(α[ind])
        wtmp = copy(w[ind])
        g, gp, wp = (copy(wtmp) for _ = 1:3)
        n_ind = size(ind, 1)
        # loop through batches
        @showprogress 0.1 "epoch $(iter-1) " for b ∈ 1:num_batches
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
        println("epoch ", iter-1, " done. incr ", incr)
        if abs(incr) < rtol || iter == maxiter
            if iter == maxiter
                @warn "Not converged after $(maxiter) iterations."
            end
            H = zeros(T, n_ind, n_ind)
            for b ∈ 1:num_batches
                if b != num_batches
                    Xtmp = @view X[(b-1)*BatchSize+1:b*BatchSize, ind]
                    mul!(a1, Xtmp, wtmp)
                    y1 .= logistic.(a1)
                    H .+= WoodburyInv!(
                        αtmp,
                        Diagonal(sqrt.(y1 .* (1 .- y1))) * Xtmp
                    )
                else
                    Xtmp = @view X[(b-1)*BatchSize+1:end, ind]
                    mul!(a2, Xtmp, wtmp)
                    y2 .= logistic.(a2)
                    H .+= WoodburyInv!(
                        αtmp,
                        Diagonal(sqrt.(y2 .* (1 .- y2))) * Xtmp
                    )
                end
            end
            H ./= num_batches
            println("done.")
            return BnRVModel(wtmp, convert(Array{T}, Symmetric(H)), ind)
        end
    end
end

function Logit!(
    w::AbstractVector{T}, α::AbstractVector{T},
    X::AbstractMatrix{T}, Xt::AbstractMatrix{T},
    t::AbstractVector{T}, tol::T,
    maxiter::Int, a::AbstractVector{T},
    y::AbstractVector{T}, g::AbstractVector{T},
    gp::AbstractVector{T}, wp::AbstractVector{T},
    ϵ::T=convert(T, 1e-8)
) where T<:Real
    n = size(X, 1)
    d = size(X, 2)
    #gp = zeros(T, d)
    mul!(a, X, w)
    llhp = -Inf32
    #wp = similar(w)
    y .= logistic.(a)
    #y = round.(y, digits=4)
    r = [0.0001f0]
    for iter = 2:maxiter
        copyto!(gp, g)
        mul!(g, Xt, t .- y)
        g .-= α .* w
        copyto!(wp, w)
        w .+= g .* r
        mul!(a, X, w)
        llh = -sum(log1pexp.((1 .- 2 .* t) .* a)) - sum(α .* w .^ 2)/2
        while !(llh - llhp > 0) & !(abs(llh - llhp) < tol)
            r ./= 2
            w .= wp .+ g .* r
            mul!(a, X, w)
            llh = -sum(log1pexp.((1 .- 2 .* t) .* a)) - sum(α .* w .^ 2)/2
        end

        y .= logistic.(a)
        #y = round.(y, digits=4)
        if abs(llh - llhp) < tol || iter == maxiter
            llh += sum(log.(α))/2 - d*log(2π)/2
            WoodburyInv!(g, α, Diagonal(sqrt.(y .* (1 .- y))) * X)
            α .= (1 .- α .* g) ./ (w .^ 2 .+ ϵ)
            if iter == maxiter
                @warn "Not converged in finding the posterior of wh."
            end
            return llh
        end
        #@debug "r1" abs(sum((w .- wp) .* (g .- gp)))
        #@debug "r2" (sum((g .- gp) .^ 2) + 1e-4)
        #@debug "g - gp, w - wp" g .- gp w .- wp
        r .= abs(sum((w .- wp) .* (g .- gp))) / (sum((g .- gp) .^ 2) + ϵ)
        #@debug "r" r
        llhp = llh
    end
end

"""
train + predict
"""
function RVM!(
    model::BnRVModel{T}, XL::AbstractMatrix{T},
    t::AbstractVector{T}, XLtest::AbstractMatrix{T},
    α::AbstractVector{T}, β::AbstractVector{T};
    rtol::T=convert(T, 1e-5), atol::T=convert(T, 1e-8),
    maxiter::Int=10000, n_samples::Int=5000,
    BatchSize::Int=size(XL, 1), ϵ::T=convert(T, 1e-8)
) where T<:Real
    n, d = size(XL)
    # should add more validity checks
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))
    num_batches = convert(Int, round(n / BatchSize))
    # preallocate type-II likelihood (evidence) vector
    llh = zeros(T, maxiter)
    llh[1] = -Inf32
    # remove zero columns
    ind_nonzero = findall(
        in(findall(x -> x > 0.001f0, std(XL, dims=1)[:])),
        model.ind
    )
    ind_h = model.ind[ind_nonzero]
    # prune wh and H
    wh = model.w[ind_nonzero]
    H = model.H[ind_nonzero, ind_nonzero]
    #@show ind_h ind
    #@show ind_nonzero
    if n_samples == 1
        whsamples = wh[:, :]
        #β = diag(LinearAlgebra.inv!(cholesky(model.H)))
        #β = β[ind_nonzero]
    else
        println("Generate posterior samples of wh...")
        whsamples = rand(
            MvNormal(
                wh,
                Symmetric(H)
            ),
            n_samples
        )
    end
    @info "whsamples" whsamples
    # remove irrelevant columns
    XL = XL[:, ind_h]
    #wl = zeros(T, size(ind_h, 1))
    β = β[ind_h]
    println("Setup done.")
    for iter ∈ 2:maxiter
        # remove irrelavent features
        ind_l = findall(β .< (1/rtol)) # optional
        n_ind = size(ind_l, 1)
        if n_ind == 0
            return predict(model, XLtest)
        end
        n_ind = size(ind_l, 1)
        βtmp = copy(β[ind_l])
        whtmp = copy(whsamples[ind_l, :])
        XLtmp = copy(XL[:, ind_l])
        Q, R = qr(randn(T, n_ind, n_ind))
        g = whtmp |> eachcol |>
        Map(
            x -> cal_rotation(
                x, Q, βtmp, XLtmp, transpose(XLtmp),
                t, atol, maxiter, ϵ, BatchSize
            )
        ) |> Broadcasting() |> Folds.sum
        g ./= n_samples
        llh[iter] = g[end] + sum(log.(βtmp)) / 2 - n_ind * log(2π) / 2
        β[ind_l] .= @views (1 .- βtmp .* g[n_ind+1:2n_ind]) ./ (g[1:n_ind] .+ ϵ)
        incr = (llh[iter] - llh[iter-1]) / llh[iter-1]
        println("iteration ", iter-1, " done. incr ", incr)
        if abs(incr) < rtol || iter == maxiter
            if iter == maxiter
                @warn "Not converged after $(maxiter) iterations.
                    Results might be inaccurate."
            end
            predictions = whtmp |> eachcol |>
            Map(
                x -> cal_rotation(
                    x, Q, βtmp, XLtmp, transpose(XLtmp),
                    t, XLtest[:, ind_h[ind_l]], atol, maxiter,
                    ϵ, BatchSize
                )
            ) |> Broadcasting() |> Folds.sum
            return predictions ./ n_samples
        end
    end
end

function cal_rotation(
    wh::AbstractVector{T}, Uinit::AbstractMatrix{T},
    α::AbstractVector{T}, X::AbstractMatrix{T}, Xt::AbstractMatrix{T},
    t::AbstractVector{T}, tol::T, maxiter::Int,
    ϵ::T=convert(T, 1e-8), bs::Int=size(X, 1), is_final=false
) where T <: Real
    n, d = size(X)
    #q, r = qr(randn(d, d))
    U, Up = (copy(Uinit) for _ = 1:2)
    #U, Up = (Diagonal(ones(T, d)) for _ = 1:2)
    g, gp = (zeros(T, d, d) for _ = 1:2)
    #@debug "U" U' * U size(U)
    #@debug "g" size(g)
    #β = [0.9f0]
    a, y = (Vector{T}(undef, n) for _ = 1:2)
    mul!(a, X, U' * wh)
    y = logistic.(a)
    llhp = Inf
    η = [5f-7]
    for iter = 2:maxiter
        for nn = 1:Int(round((n/bs)))
            @views mul!(
                g, wh[:, :],
                (y[1 + bs*(nn-1):bs*nn] .- t[1 + bs*(nn-1) : bs*nn])' *
                X[1 + bs*(nn-1) : bs*nn, :]
            )
            #
            g .+= Diagonal(α) * U' * wh * wh'
            g .= g * U' .- U * g'
            #g .-= U * transpose(g) * U
            copyto!(Up, U)
            mul!(U, I - η .* g, Up)
            ldiv!(qr!(I + η .* g), U)
            #U .-= η .* g
            mul!(a, X, U' * wh)
            y .= logistic.(a)
        end
        llh = sum(log1pexp.((1 .- 2 .* t) .* a))
        @debug "llh-llhp" llh-llhp
        #@debug "U" U' * U
        @debug "sum(g .^ 2)" sum(g.^2)
        @debug "sum(g .^ 2)" sum((g .- gp).^2)
        if sum((g .- gp).^2) < tol || iter == maxiter
            if iter == maxiter
                @warn "Not conerged after $(maxiter) steps."
            end
            # calculate diagonal of invH
            WoodburyInv!(view(g, :, 1), α, Diagonal(sqrt.(y .* (1 .- y))) * X)
            wl = U' * wh  #Uinit .= U
            return vcat(wl .^ 2, view(g, :, 1), -llh -sum(α.*wl.^2)/2)
        else
            llhp = llh
            copyto!(gp, g)
        end
    end
end

function cal_rotation(
    wh::AbstractVector{T}, Uinit::AbstractMatrix{T},
    α::AbstractVector{T}, X::AbstractMatrix{T}, Xt::AbstractMatrix{T},
    t::AbstractVector{T}, Xtest::AbstractMatrix{T}, tol::T, maxiter::Int,
    ϵ::T=convert(T, 1e-8), bs::Int=size(X, 1)
) where T <: Real
    n, d = size(X)
    U, Up = (copy(Uinit) for _ = 1:2)
    #U, Up = (Diagonal(ones(T, d)) for _ = 1:2)
    g, gp = (zeros(T, d, d) for _ = 1:2)
    a, y = (Vector{T}(undef, n) for _ = 1:2)
    mul!(a, X, U' * wh)
    y = logistic.(a)
    llhp = Inf
    η = [5f-7]
    for iter = 2:maxiter
        for nn = 1:Int(round((n/bs)))
            @views mul!(
                g, wh[:, :],
                (y[1 + bs*(nn-1):bs*nn] .- t[1 + bs*(nn-1) : bs*nn])' *
                X[1 + bs*(nn-1) : bs*nn, :]
            )
            #
            g .+= Diagonal(α) * U' * wh * wh'
            g .= g * U' .- U * g'
            #g .-= U * transpose(g) * U
            copyto!(Up, U)
            mul!(U, I - η .* g, Up)
            ldiv!(qr!(I + η .* g), U)
            #U .-= η .* g
            mul!(a, X, U' * wh)
            y .= logistic.(a)
        end
        llh = sum(log1pexp.((1 .- 2 .* t) .* a))
        @debug "llh-llhp" llh-llhp
        #@debug "U" U' * U
        @debug "g" g
        @debug "sum(g .^ 2)" sum((g .- gp).^2)
        if sum((g .- gp).^2) < tol || iter == maxiter
            if iter == maxiter
                @warn "Not conerged after $(maxiter) steps."
            end
            Σ = WoodburyInv!(α, Diagonal(sqrt.(y .* (1 .- y))) * X)
            return predict(Xtest, U'*wh, Σ)
        else
            llhp = llh
            copyto!(gp, g)
        end
    end
end
