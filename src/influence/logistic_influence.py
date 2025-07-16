import numpy as np
from .base import InfluenceFunctionBase


class LogisticInfluence(InfluenceFunctionBase):
    """
    Influence function calculator for Logistic Regression models.
    
    Implements influence calculation using the Hessian-based approach. Measures how much each training point
    affects the model's predictions on test data by approximating parameter changes if the training point was removed.
    
    Attributes:
        model: Trained logistic regression model
        X_train: Training feature matrix (n_samples, n_features)
        y_train: Training label vector (n_samples,)
        hessian_inv: Inverse of the regularized Hessian matrix (n_features, n_features)
    """
    
    def __init__(self, model, X_train, y_train):
        """
        Initialize influence calculator for logistic regression.
        
        Args:
            model: Pretrained logistic regression model
            X_train: Training data features
            y_train: Training data labels
        """
        super().__init__(model, loss_fn=None)
        self.X_train = X_train
        self.y_train = y_train
        self.hessian_inv = self._get_inv_hessian()

    def _get_inv_hessian(self):
        """
        Compute inverse Hessian for logistic regression log-loss.
        
        Hessian = XᵀSX where S is diagonal matrix of probabilities.
        Adds regularization for numerical stability.
        
        Returns:
            Inverse of regularized Hessian matrix
        """
        probs = self.model.predict_proba(self.X_train)[:, 1]  # P(y = 1 | x)
        S = np.diag(probs * (1 - probs))
        H = self.X_train.T @ S @ self.X_train
        H_reg = H + 1e-6 * np.eye(H.shape[0])
        return np.linalg.inv(H_reg)

    def _grad_loss(self, x, y):
        """Compute gradient of logistic loss at single data point"""
        pred = self.model.predict_proba(x.reshape(1, -1))[0, 1]
        return (pred - y) * x

    def get_influence(self, x_test, y_test):
        """
        Calculate influence of all training points on single test point.
        
        Influence = -∇L(test)ᵀ H⁻¹ ∇L(train_i)
        
        Args:
            x_test: Test point features
            y_test: Test point label
            
        Returns:
            List of influence scores for each training point
        """
        grad_test = self._grad_loss(x_test, y_test)
        influences = []
        for i in range(len(self.X_train)):
            grad_i = self._grad_loss(self.X_train[i], self.y_train[i])
            influence = -grad_test.T @ self.hessian_inv @ grad_i
            influences.append(influence)
        return influences

    def average_influence(self, X_test, y_test):
        """
        Compute average influence over multiple test points.
        
        Args:
            X_test: Test feature matrix (n_test, n_features)
            y_test: Test label vector (n_test,)
            
        Returns:
            Average influence scores for each training point
        """
        total_influence = np.zeros(len(self.X_train))
        for i in range(len(X_test)):
            influence_i = self.get_influence(X_test[i], y_test[i])
            total_influence += influence_i
        return total_influence / len(X_test)

class LinearSVMInfluence(InfluenceFunctionBase):
    """
    Influence function calculator for linear SVM models.
    
    Uses approximate Hessian for hinge loss. Measures training point influence using
    subgradients and L2-regularized Hessian approximation.
    
    Attributes:
        model: Trained linear SVM model
        X_train: Training feature matrix (n_samples, n_features)
        y_train: Training label vector (n_samples,)
        reg: Regularization strength (λ)
        hessian_inv: Approximate inverse Hessian (λ⁻¹ I)
    """
    
    def __init__(self, model, X_train, y_train, reg=1.0):
        """
        Initialize influence calculator for linear SVM.
        
        Args:
            model: Pretrained linear SVM model
            X_train: Training data features
            y_train: Training data labels
            reg: Regularization strength (default: 1.0)
        """
        super().__init__(model, loss_fn=None)
        self.X_train = X_train
        self.y_train = y_train
        self.reg = reg
        self.hessian_inv = self._get_inv_hessian()

    def _grad_loss(self, x, y):
        """Compute subgradient of hinge loss at single data point"""
        margin = y * np.dot(self.model.coef_, x)
        if margin >= 1:
            return np.zeros_like(x)
        else:
            return -y * x

    def _get_inv_hessian(self):
        """
        Approximate inverse Hessian for linear SVM.
        
        Uses simplified form H⁻¹ ≈ (λI)⁻¹ since exact Hessian
        is undefined for hinge loss. More accurate approximations
        could use XᵀDX + λI where D is diagonal matrix.
        
        Returns:
            Approximate inverse Hessian matrix
        """
        d = self.X_train.shape[1]
        return np.linalg.inv(self.reg * np.eye(d))

    def get_influence(self, x_test, y_test):
        """
        Calculate influence of all training points on single test point.
        
        Args:
            x_test: Test point features
            y_test: Test point label
            
        Returns:
            List of influence scores for each training point
        """
        grad_test = self._grad_loss(x_test, y_test)
        influences = []
        for i in range(len(self.X_train)):
            grad_i = self._grad_loss(self.X_train[i], self.y_train[i])
            infl = - grad_test.T @ self.hessian_inv @ grad_i
            influences.append(infl)
        return influences

    def average_influence(self, X_test, y_test):
        """
        Compute average influence over multiple test points.
        
        Args:
            X_test: Test feature matrix (n_test, n_features)
            y_test: Test label vector (n_test,)
            
        Returns:
            Average influence scores for each training point
        """
        total_influence = np.zeros(len(self.X_train))
        for i in range(len(X_test)):
            influence_i = self.get_influence(X_test[i], y_test[i])
            total_influence += influence_i
        return total_influence / len(X_test)
