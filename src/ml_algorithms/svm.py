import numpy as np

class SVM:

    def __init__(self, learning_rate: float = 0.01, lambda_param: float = 0.01, n_iters: int = 1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_norm = np.where(y <= 0, -1, 1) # only works for 2 classes though
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0

        for _ in range(self.n_iters):
            for i, x_i in enumerate(X):
                condition = y_norm[i]  * np.dot(x_i, self.weights) - self.bias >= 1
                if condition:
                    self.weights -= 2 * self.lr * self.lambda_param * self.weights
                else:
                    self.weights -= 2 * self.lr * self.lambda_param * self.weights - np.dot(x_i, y_norm[i])
                    self.bias -= self.lr * y_norm[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(np.dot(X, self.weights) - self.bias)
