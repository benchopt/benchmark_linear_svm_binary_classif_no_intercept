from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from thundersvm import SVC


class Solver(BaseSolver):
    name = "thundersvm"

    install_cmd = "conda"
    requirements = [
        "pip:https://github.com/Xtra-Computing/thundersvm/blob/"
        "d38af58e0ceb7e5d948f3ef7d2c241ba50133ee6/python/dist/"
        "thundersvm-cpu-0.2.0-py3-none-linux_x86_64.whl?raw=true"]

    def set_objective(self, X, y, C):
        self.X, self.y, self.C = X, y, C
        self.clf = SVC(C=self.C, tol=1e-12, n_jobs=1)

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.flatten()
