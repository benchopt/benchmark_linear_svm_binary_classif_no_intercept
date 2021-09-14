from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import issparse
    from numba import njit

if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


@njit
def clip(x, C):
    if x > C:
        return C
    if x < 0:
        return 0
    return x


class Solver(BaseSolver):
    name = "CD"

    install_cmd = 'conda'
    requirements = ['numba']

    stop_strategy = 'iteration'
    support_sparse = True

    # We increase patience to give enough time to the dual solver
    # to actually decrease the primal objective
    stopping_criterion = SufficientProgressCriterion(eps=1e-4, patience=10)

    def set_objective(self, X, y, C):
        self.X, self.y, self.C = X, y, C

        if issparse(X):
            self.X = X
        else:
            # use Fortran order to compute gradient on contiguous rows
            self.X = np.ascontiguousarray(X)

        # Make sure we cache the numba compilation.
        self.run(1)

    def run(self, n_iter):

        if issparse(self.X):
            yX = self.X.multiply(self.y[:, None]).tocsr()
            L = np.array((self.X.multiply(self.X)).sum(axis=1)).squeeze()
            beta = self.sparse_cd(
                yX.data, yX.indices, yX.indptr, self.C, L, yX.shape[1], n_iter
            )
        else:
            yX = self.X * self.y[:, None]
            L = (self.X ** 2).sum(axis=1)
            beta = self.cd(yX, self.C, L, n_iter)

        self.beta = np.asarray(beta).flatten()

    @staticmethod
    @njit
    def cd(yX, C, L, n_iter):
        n_samples, n_features = yX.shape

        mu = np.zeros(n_samples)
        w = np.zeros(n_features)

        for _ in range(n_iter):
            for j in range(n_samples):
                if L[j] == 0.:
                    continue

                alpha_j = 1 / L[j]
                old_mu_j = mu[j]
                grad_f_j = yX[j] @ w - 1
                new_mu_j = clip(old_mu_j - alpha_j * grad_f_j, C)

                if new_mu_j != old_mu_j:
                    w += yX[j]*(new_mu_j - old_mu_j)
                    mu[j] = new_mu_j

        return w

    @staticmethod
    @njit
    def sparse_cd(yX_data, yX_indices, yX_indptr, C, L, n_features, n_iter):
        n_samples = len(yX_indptr) - 1
        w = np.zeros(n_features)
        mu = np.zeros(n_samples)

        for _ in range(n_iter):
            for j in range(n_samples):
                if L[j] == 0.:
                    continue

                alpha_j = 1 / L[j]
                old_mu_j = mu[j]
                start, end = yX_indptr[j:j+2]

                grad_f_j = -1
                for ind in range(start, end):
                    grad_f_j += yX_data[ind] * w[yX_indices[ind]]

                new_mu_j = clip(old_mu_j - alpha_j * grad_f_j, C)

                if new_mu_j != old_mu_j:
                    for ind in range(start, end):
                        w[yX_indices[ind]] += (
                            yX_data[ind]*(new_mu_j - old_mu_j)
                        )
                    mu[j] = new_mu_j

        return w

    def get_result(self):
        return self.beta
