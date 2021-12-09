from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    from flashcd.estimators import SVC


class Solver(BaseSolver):
    name = 'flashcd'
    stop_strategy = 'iteration'

    install_cmd = 'conda'
    requirements = ['flashcd']
    support_sparse = True

    stopping_criterion = SufficientProgressCriterion(eps=1e-4, patience=10)

    def set_objective(self, X, y, C):
        self.X, self.y, self.C = X, y, C

        self.clf = SVC(C=self.C, tol=1e-12)
        # Make sure we cache the numba compilation.
        self.run(1)

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.flatten()
