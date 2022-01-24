import warnings
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numpy.linalg import norm
from sklearn.svm import LinearSVC

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

C = 0.1
clf = LinearSVC(C=C, penalty='l2', dual=True, fit_intercept=False, tol=1e-12,
                loss='hinge', random_state=0)


@njit
def loss(w, yX, C):
    return np.maximum(0, 1 - yX @ w).sum() + norm(w) ** 2 / (2 * C)


@njit
def clip(x, C):
    if x > C:
        return C
    if x < 0:
        return 0
    return x


@njit
def cd_dual(yX, C, n_iter):
    """Solve problem with coordinate descent in the dual, ie minimize:
    norm(yX @ mu) ** 2 / 2 + mu.sum()    subject to 0 <= mu <= C

    with link equation w = yX @ mu  (=sum_i mu_i * y_i * X[i]"""
    n_samples, n_features = yX.shape
    lipschitz = np.zeros(n_samples)
    for j in range(n_samples):
        lipschitz[j] = norm(yX[j]) ** 2

    mu = np.zeros(n_samples)
    w = np.zeros(n_features)
    losses = np.zeros(n_iter)
    for it in range(n_iter):
        for j in range(n_samples):
            old_mu_j = mu[j]
            grad_f_j = yX[j] @ w - 1
            new_mu_j = clip(old_mu_j - grad_f_j / lipschitz[j], C)

            if new_mu_j != old_mu_j:
                w += yX[j] * (new_mu_j - old_mu_j)
                mu[j] = new_mu_j
        losses[it] = loss(w, yX, C)

    return w, losses


fig, axarr = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for idx, (n_samples, n_features) in enumerate([[100, 500], [500, 100]]):
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    y = X @ np.random.randn(n_features) + 0.1 * np.random.randn(n_samples)
    y = 2 * (y > 0) - 1
    yX = X * y[:, None]

    max_iter = 3000 if n_samples > n_features else 300
    losses_sk = []
    iter_sk = np.geomspace(1, max_iter, num=50).astype(int)
    for n_iter in iter_sk:
        clf.max_iter = n_iter
        clf.fit(X, y)
        w = clf.coef_[0]
        losses_sk.append(loss(w, yX, C))
    losses_sk = np.array(losses_sk)

    w, losses_cd = cd_dual(yX, C, max_iter)
    loss_min = min(losses_cd.min(), losses_sk.min())

    axarr[idx].semilogy(iter_sk, losses_sk - loss_min, label='sklearn')
    axarr[idx].semilogy(losses_cd - loss_min, label='dual coordinate descent')
    axarr[idx].set_xlabel("iteration")
    axarr[idx].set_ylabel("loss - min loss")
    axarr[idx].set_title(f'n_samples, n_features=({n_samples}, {n_features})')
    axarr[idx].legend()
plt.show(block=False)


# no intercept is fitted
np.testing.assert_allclose(clf.decision_function(X), X @ clf.coef_[0])
