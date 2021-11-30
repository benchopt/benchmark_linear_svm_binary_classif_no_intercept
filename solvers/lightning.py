from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from lightning.classification import LinearSVC


class Solver(BaseSolver):
    name = "lightning"
    stop_strategy = "iteration"

    install_cmd = "conda"
    requirements = ["sklearn-contrib-lightning"]

    def set_objective(self, X, y, C):
        self.X, self.y, self.C = X, y, C
        self.clf = LinearSVC(warm_start=False, tol=0, loss="hinge")

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.ravel()
