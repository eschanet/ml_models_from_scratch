import numpy as np


class Perceptron:

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.act_func = self._unit_step_func
        self.weights = None
        self.bias = None
    
    def _unit_step_func(self,x: np.ndarray) -> np.array:
        return np.where(x >= 0, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0
        y_norm = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for i, x_i in enumerate(X):
                y_pred = self.act_func(
                    np.dot(x_i, self.weights) + self.bias
                )
                dw = self.lr * (y_norm[i] - y_pred)
                self.weights += dw * x_i
                self.bias += dw


    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.act_func(
            np.dot(X, self.weights) + self.bias
        )
