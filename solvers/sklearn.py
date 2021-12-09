import warnings
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import LinearSVC


class Solver(BaseSolver):
    name = 'sklearn'

    install_cmd = 'conda'
    requirements = ['scikit-learn']
    stopping_criterion = SufficientProgressCriterion(eps=1e-4, patience=10)

    parameters = {
        'solver': ['liblinear'],
    }
    parameter_template = "solver={solver}"

    def set_objective(self, X, y, C):
        self.X, self.y, self.C = X, y, C

        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.clf = LinearSVC(C=self.C, penalty='l2', dual=True,
                             fit_intercept=False, tol=1e-12,
                             loss='hinge')

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.flatten()
