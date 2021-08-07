import numpy as np
from typing import List


class LinearRegression:
    def __init__(self, lr: float = 0.001, n_iters: int = 1000):
        """Initialize linear regressor

        Args:
            lr (float, optional): Learning rate. Defaults to 0.001.
            n_iters (int, optional): Number of optimisation iterations. Defaults to 1000.
        """ 

        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to some data.

        Args:
            X (np.ndarray): Independent variables to fit the model to.
            y (np.ndarray): Dependent variables.
        """        
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # calculate gradient
            dw = (1.0 / self.n_iters) * np.dot(X.T, (y_predicted - y))
            db = (1.0 / self.n_iters) * np.sum(y_predicted - y)

            # update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> List[int]:
        """Predict the dependent value for some data.

        Args:
            X (np.ndarray): Independent variables to predict dependent value for.

        Returns:
            List[int]: Predictions for each data point.
        """        
        return np.dot(X, self.weights) + self.bias
