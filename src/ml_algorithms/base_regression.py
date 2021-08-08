from abc import ABC, abstractmethod
import numpy as np

class BaseRegression(ABC):

    def __init__(self, lr: float = 0.001, n_iters: int = 1000):
        """Initialize base regressor

        Args:
            lr (float, optional): Learning rate. Defaults to 0.001.
            n_iters (int, optional): Number of optimisation iterations. Defaults to 1000.
        """ 
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.array):
        """Fit the model to some data.

        Args:
            X (np.ndarray): Independent variables to fit the model to.
            y (np.ndarray): Dependent variables.
        """    
        n_samples, n_features = X.shape
        
        # initialize parameters
        self.weights, self.bias = np.zeros(n_features), 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = self._approximation(X, self.weights, self.bias)

            dw = (1.0 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1.0 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X: np.ndarray):
        """Predict the dependent value for some data.

        Args:
            X (np.ndarray): Independent variables to predict dependent value for.

        Returns:
            List[int]: Predictions for each data point.
        """        
        return self._predict(X, self.weights, self.bias)

    @abstractmethod
    def _approximation(self, X: np.ndarray, w: np.array, b: float):
        pass

    @abstractmethod
    def _predict(self, X: np.ndarray, w: np.array, b: float):
        pass
