#!/usr/bin/env python3
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal


def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    theta_ = np.array([tau, mu1, sigma1, mu2, sigma2])

    def neg_ll(th_):
        tau, mu1, sigma1, mu2, sigma2 = th_

        L1 = tau * norm(mu1, sigma1).pdf(x)
        L2 = (1 - tau) * norm(mu2, sigma2).pdf(x)
        L = L1 + L2
        l = np.log(L,  where=L != 0)
        return -sum(l)

    res = minimize(neg_ll,
                   theta_,
                   method='Nelder-Mead',
                   options={'xatol': rtol},
                   bounds=[(0, 1),
                           (None, None),
                           (0, None),
                           (None, None),
                           (0, None)]
                   ).x

    return tuple(res)


def M(x, tau, mu1, sigma1, mu2, sigma2):

    L1 = tau * norm(mu1, sigma1).pdf(x)
    L2 = (1 - tau) * norm(mu2, sigma2).pdf(x)
    L = L1 + L2

    T1 = np.divide(L1, L, out=np.full_like(L, 0.5), where=L != 0)
    T2 = np.divide(L2, L, out=np.full_like(L, 0.5), where=L != 0)

    tau = np.sum(T1) / np.sum(T1+T2)
    mu1 = np.sum(T1 * x) / np.sum(T1)
    mu2 = np.sum(T2 * x) / np.sum(T2)
    sigma1 = np.sqrt(np.sum((x - mu1) ** 2 * T1) / np.sum(T1))
    sigma2 = np.sqrt(np.sum((x - mu2) ** 2 * T2) / np.sum(T2))
    return tau, mu1, sigma1, mu2, sigma2


def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    i_max = int(1 / rtol)
    theta_ = np.array([tau, mu1, sigma1, mu2, sigma2])

    for i in range(i_max):

        res = M(x, *theta_)

        if np.allclose(theta_, res, rtol=rtol):
            break
        else:
            theta_ = res

    return tuple(res)


def M_cluster(x, T1, T2, T3):

    tau1 = np.sum(T1) / np.sum(T1 + T2 + T3)
    tau2 = np.sum(T2) / np.sum(T1 + T2 + T3)

    # mu1_ = np.sum(np.dot(T1, x)) / np.sum(T1)
    # mu2_ = np.sum(np.dot(T2, x)) / np.sum(T2)

    mu1_ = np.zeros(4)
    mu2_ = np.zeros(4)

    for i in range(4):
        mu1_[i] = np.sum(T1 * x[:, i]) / np.sum(T1)
        mu2_[i] = np.sum(T2 * x[:, i]) / np.sum(T2)

    sigma_ = np.diag(T1 @ (x - mu1_) ** 2) / np.sum(T1)
    sigma02_ = np.diag(T3 @ x[:, 2:] ** 2) / np.sum(T3)

    return tau1, tau2,\
        mu1_[:2], mu2_[:2], mu1_[2:],\
        [sigma_[0, 0], sigma_[0, 0]],\
        [sigma_[2, 2], sigma_[2, 2]], sigma02_[0, 0]


def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2, rtol=1e-3):

    i_max = int(1 / rtol)

    for i in range(i_max):

        mu1_ = np.hstack((mu1, muv))
        mu2_ = np.hstack((mu2, muv))
        sigma_ = np.diag(np.hstack((sigmax2, sigmav2)))
        sigma02_ = np.diag([sigma02, sigma02])

        L1 = tau1 * multivariate_normal(mu1_, sigma_).pdf(x)
        L2 = tau2 * multivariate_normal(mu2_, sigma_).pdf(x)
        L3 = (1 - tau1 - tau2) * multivariate_normal([0, 0], sigma02_).pdf(x[:, 2:])
        L = L1 + L2 + L3

        T1 = np.divide(L1, L, out=np.full_like(L, 0.5), where=L != 0)
        T2 = np.divide(L2, L, out=np.full_like(L, 0.5), where=L != 0)
        T3 = np.divide(L3, L, out=np.full_like(L, 0.5), where=L != 0)

        res = M_cluster(x, T1, T2, T3)

        tau1, tau2, mu1, mu2, muv, sigmax2, sigmav2, sigma02 = res

    return res, T1, T2, T3


if __name__ == "__main__":
    x = 0.7*np.random.normal(1,4,50) + 0.3*np.random.normal(70,1,50)
    dist = norm(3, 1)
    print(max_likelihood(x, 0.5, 0, 3, 60, 1))
    print(em_double_gauss(x, 0.5, 0, 3, 50, 1))
