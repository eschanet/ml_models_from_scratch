from collections import Counter
import numpy as np
from scipy.spatial import distance

class KNN:

    def __init__(self, k: int):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x: np.ndarray) -> int:
        distances = [distance.minkowski(x, x_train, p=2) for x_train in self.X_train]
        k_neighbor_labels = [self.y_train[i] for i in np.argsort(distances)[: self.k]]
        return Counter(k_neighbor_labels).most_common(1)[0][0]
