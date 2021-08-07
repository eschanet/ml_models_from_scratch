import numpy as np
from typing import List

class LogisticRegression:

    def __init__(self, lr: float = 0.001, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.array):
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1.0 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1.0 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.array:
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)

        return np.array([1 if p > 0.05 else 0 for p in y_predicted])

    def _sigmoid(self, x: np.array) -> np.array:
        return 1.0 / (1 + np.exp(-x))
