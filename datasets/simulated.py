from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from benchopt.datasets import make_correlated_data


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_samples, n_features': [
            (300, 1000),
            (1000, 300)]
    }

    def __init__(self, n_samples=10, n_features=50, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):
        X, y, _ = make_correlated_data(
            n_samples=self.n_samples, n_features=self.n_features,
            random_state=self.random_state
        )
        y = 2*(y > 0) - 1

        data = dict(X=X, y=y)

        return self.n_features, data
