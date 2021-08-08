import numpy as np
from typing import List

from .base_regression import BaseRegression

class LogisticRegression(BaseRegression):

    def _approximation(self, X: np.ndarray, w: np.array, b: float):
        linear_model = np.dot(X, w) + b
        return self._sigmoid(linear_model)

    def _predict(self, X: np.ndarray, w: np.array, b: float):
        linear_model = np.dot(X, w) + b
        y_predicted = self._sigmoid(linear_model)
        return np.array([1 if p > 0.05 else 0 for p in y_predicted])

    def _sigmoid(self, x: np.array) -> np.array:
        return 1.0 / (1 + np.exp(-x))
