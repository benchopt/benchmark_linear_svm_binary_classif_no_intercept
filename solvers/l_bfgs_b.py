import numpy as np
from numpy.linalg import norm
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import issparse

from benchopt import BaseSolver


class Solver(BaseSolver):
    name = "L-BFGS-B"

    stop_strategy = 'iteration'
    support_sparse = True

    def set_objective(self, X, y, C):
        self.X, self.y, self.C = X, y, C

    def run(self, n_iter):
        X, y, C = self.X, self.y, self.C
        n_samples = X.shape[0]
        if issparse(X):
            yX = X.multiply(y[:, None])
        else:
            yX = X * y[:, None]

        def f(mu):
            Gmu = mu @ yX
            return 0.5 * norm(Gmu) ** 2 - np.sum(mu)

        def gradf(mu):
            grad = (mu @ yX) @ yX.T
            grad -= 1
            return grad

        mu0 = np.zeros(n_samples)
        bounds = n_samples * [(0, C)]  # set box constraints
        mu_hat, _, _ = fmin_l_bfgs_b(f, mu0, gradf, bounds=bounds,
                                     pgtol=0., factr=0., maxiter=n_iter)
        beta = (mu_hat * y) @ X
        self.beta = np.asarray(beta).flatten()

    def get_result(self):
        return self.beta
