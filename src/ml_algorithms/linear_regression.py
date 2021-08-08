import numpy as np
from typing import List

from .base_regression import BaseRegression

class LinearRegression(BaseRegression):

    def _approximation(self, X: np.ndarray, w: np.array, b: float):
        return np.dot(X, w) + b

    def _predict(self, X: np.ndarray, w: np.array, b: float) -> List[int]:
        return np.dot(X, w) + b
