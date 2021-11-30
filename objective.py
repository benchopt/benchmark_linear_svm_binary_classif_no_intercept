import numpy as np
from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "SVM Binary Classification (no intercept)"

    parameters = {
        'C': [0.01],
    }

    def __init__(self, C=1.):
        self.C = C

    def set_data(self, X, y):
        self.X, self.y = X, y

    def compute(self, beta):
        loss = self.C * np.sum(
            np.maximum(1 - self.y * (self.X @ beta), 0.)
        )
        pen = 0.5 * np.dot(beta, beta)
        return loss + pen

    def to_dict(self):
        return dict(X=self.X, y=self.y, C=self.C)
