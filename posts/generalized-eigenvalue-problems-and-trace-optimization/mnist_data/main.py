import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
from julia import Main
from itertools import product
from math import exp


PIXEL_WIDTH = 28
IMAGE_SIZE = PIXEL_WIDTH * PIXEL_WIDTH

df = pd.read_csv("train.csv")
digits = {}

for (d, g) in df.groupby("label"):
    X = np.array(df)
    g.drop("label", axis=1, inplace=True)

    digits[d] = np.array(g)


def estimate_cov(X):
    """X should be an n x d array with n data points in dimension d."""
    n, d = X.shape
    return X.T @ X / n


def error_variance(r, i, j, u, v):
    return exp(-0.5 * r * (abs(u - i) ** 2 + abs(v - j) ** 2))


def k_to_ij(k):
    i = k // PIXEL_WIDTH
    j = k % PIXEL_WIDTH
    return i, j


def ij_to_k(i, j):
    return i * PIXEL_WIDTH + j


def optimal_snr_pca(Sigma, Omega, k=2, eps=1e-6):
    n = Sigma.shape[0]
    eig_subset = [n - k, n - 1]

    def M(s):
        return Sigma - s * Omega

    def S(s):
        return np.sum(la.eigvalsh(M(s), subset_by_index=eig_subset))

    def V(s):
        _, v = la.eigh(M(s), subset_by_index=eig_subset)
        return v

    s_max = float(la.eigvalsh(Sigma, Omega, subset_by_index=[n - 1, n - 1]))

    s_star = bisection(S, s_max, eps)
    V_star = V(s_star)
    return V_star, s_star


def bisection(S, s_max, eps=1e-6):
    s_left = 0.0
    s_right = s_max
    while abs(s_right - s_left) > eps:
        s = s_left + (s_right - s_left) / 2
        if S(s) > 0:
            s_left = s
        else:
            s_right = s
    return s_left


digits_cov = {digit: estimate_cov(digits[digit]) for digit in digits}
Sigma = sum(cov for cov in digits_cov.values()) / len(digits_cov)

Omega = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
for k1, k2 in product(range(IMAGE_SIZE), range(IMAGE_SIZE)):
    Omega[k1, k2] = error_variance(0.1, *k_to_ij(k1), *k_to_ij(k2))
Omega = Omega + 0.01 * np.eye(IMAGE_SIZE)

for d, D in digits.items():
    print(d)
    V_star, s_star = optimal_snr_pca(digits_cov[d], Omega, k=2, eps=1e-6)
    P = D @ V_star
    plt.scatter(P[:, 0], P[:, 1], label=d, alpha=0.25)
plt.legend()
plt.show()
