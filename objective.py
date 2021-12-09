import numpy as np
from benchopt import BaseObjective
from scipy.sparse import issparse
import scipy


class Objective(BaseObjective):
    name = "SVM Binary Classification (no intercept)"

    parameters = {
        'C': [0.1],
    }

    def __init__(self, C=1.):
        self.C = C

    def set_data(self, X, y):
        self.X, self.y = X, y

    def compute(self, beta):
        beta = np.array(beta).flatten()
        XTybeta = np.multiply(self.y, (self.X @ beta))
        loss = self.C * np.sum(
            np.maximum(1 - XTybeta, 0.)
        )
        pen = 0.5 * np.dot(beta, beta)
        p_obj = loss + pen

        # computation of dual point to compute dual gap
        # Solving a linear system for the points that are
        # inside the interval ]0, C[
        dual_point = np.zeros(len(self.y))
        support = np.isclose(np.multiply(self.y, (self.X @ beta)), 1.0)
        maskC = XTybeta < 1 - 1e-8
        dual_point[maskC] = self.C
        if issparse(self.X):
            yXT = (self.X.multiply(self.y[:, None])).T.tocsc()
            if support.sum() != 0:
                b = beta[:, None] - self.C * (yXT[:, maskC]).sum(axis=1)
                b = np.array(b)[:, 0]
                dual_point[support] = scipy.sparse.linalg.lsqr(
                    yXT[:, support], b, atol=1e-03, btol=1e-03)[0]
        else:
            yXT = (self.X * self.y[:, None]).T
            if support.sum() != 0:
                dual_point[support] = np.linalg.lstsq(
                    yXT[:, support],
                    beta - self.C * (yXT[:, maskC]).sum(axis=1),
                    rcond=None)[0]
        dual_point[dual_point < 0] = 0.0
        dual_point[dual_point > self.C] = self.C

        d_obj = ((yXT @ dual_point) ** 2).sum() / 2 - np.sum(dual_point)
        return dict(value=p_obj,
                    sv_size=(dual_point != 0).sum(),
                    duality_gap=p_obj + d_obj,)

    def to_dict(self):
        return dict(X=self.X, y=self.y, C=self.C)
