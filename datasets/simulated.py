from benchopt import BaseDataset, safe_import_context
from sklearn.preprocessing import StandardScaler
with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_samples, n_features': [
            (500, 100),
            (100, 500)]
    }

    def __init__(self, n_samples=10, n_features=50, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randn(self.n_samples)
        y = np.sign(y)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        data = dict(X=X, y=y)

        return self.n_features, data
