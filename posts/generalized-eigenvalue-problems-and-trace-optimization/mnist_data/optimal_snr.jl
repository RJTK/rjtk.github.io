module OptimalSnr
export optimal_snr_pca

using LinearAlgebra;

function snr_pca(Σ, Ω, V)
    return tr(V' * Σ * V) / tr(V' * Ω * V)
end

function optimal_snr_pca(Σ, Ω, k=2, eps=1e-6)
    n = size(Σ)[1]
    S = s -> (sum ∘ eigvals)(Σ - s * Ω, n - k + 1 : n)
    V = s -> eigvecs(Σ - s * Ω)[:, n - k + 1 : n]
    s_max = eigvals(Σ, Ω)[n]

    s_star = bisection(S, s_max, eps)
    V_star = V(s_star)
    return V_star, s_star
end

function bisection(S, s_max, eps=1e-6)
    s_left = 0.0
    s_right = s_max
    while abs(s_right - s_left) > eps
        s = s_left + (s_right - s_left) / 2
        if S(s) > 0
            s_left = s
        else
            s_right = s
        end
    end
    return s_left
end
end
