import numpy as np
from .base import InfluenceFunctionBase


class LogisticInfluence(InfluenceFunctionBase):
    def __init__(self, model, X_train, y_train):
        super().__init__(model, loss_fn=None)
        self.X_train = X_train
        self.y_train = y_train
        self.hessian_inv = self._get_inv_hessian()

    # calculate Hessian of log-loss function for Logistic Regression, i.e. sum of pi
    def _get_inv_hessian(self):
        probs = self.model.predict_proba(self.X_train)[:, 1]  # P(y = 1 | x)
        S = np.diag(probs * (1 - probs))
        H = self.X_train.T @ S @ self.X_train
        H_reg = H + 1e-6 * np.eye(H.shape[0])
        return np.linalg.inv(H_reg)

    def _grad_loss(self, x, y):
        pred = self.model.predict_proba(x.reshape(1, -1))[0, 1]
        return (pred - y) * x

    def get_influence(self, x_test, y_test):
        grad_test = self._grad_loss(x_test, y_test)
        influences = []
        for i in range(len(self.X_train)):
            grad_i = self._grad_loss(self.X_train[i], self.y_train[i])
            influence = -grad_test.T @ self.hessian_inv @ grad_i
            influences.append(influence)
        return influences

    # New: Average the influence -> do not depend only on one single output label
    def average_influence(self, X_test, y_test):
        total_influence = np.zeros(len(self.X_train))
        for i in range(len(X_test)):
            influence_i = self.get_influence(X_test[i], y_test[i])
            total_influence += influence_i
        return total_influence / len(X_test)

class LinearSVMInfluence(InfluenceFunctionBase):
    def __init__(self, model, X_train, y_train, reg=1.0):
        super().__init__(model, loss_fn=None)
        self.X_train = X_train
        self.y_train = y_train
        self.reg = reg
        self.hessian_inv = self._get_inv_hessian()

    def _grad_loss(self, x, y):
        """Subgradient of hinge loss"""
        margin = y * np.dot(self.model.coef_, x)
        if margin >= 1:
            return np.zeros_like(x)
        else:
            return -y * x

    def _get_inv_hessian(self):
        """
        For hinge loss, H ≈ λ * I (only L2 regularization)
        You can also use: H ≈ Xᵗ D X + λI for smoother approximation
        """
        d = self.X_train.shape[1]
        return np.linalg.inv(self.reg * np.eye(d))

    def get_influence(self, x_test, y_test):
        grad_test = self._grad_loss(x_test, y_test)
        influences = []
        for i in range(len(self.X_train)):
            grad_i = self._grad_loss(self.X_train[i], self.y_train[i])
            infl = - grad_test.T @ self.hessian_inv @ grad_i
            influences.append(infl)
        return influences

    def average_influence(self, X_test, y_test):
        total_influence = np.zeros(len(self.X_train))
        for i in range(len(X_test)):
            influence_i = self.get_influence(X_test[i], y_test[i])
            total_influence += influence_i
        return total_influence / len(X_test)